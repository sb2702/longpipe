// Main-side measurement of a camera frame's HIDDEN display orientation, for
// engines that don't expose VideoFrame.rotation / flip.
//
// Why: shipping Safari (iOS and macOS, through 26.x) tags camera frames with a
// rotation (and a mirror for the front camera) but keeps the pixels in sensor
// orientation, doesn't expose the tag (VideoFrame.rotation/flip sit behind
// WebKit's off-by-default WebCodecsVideoFrameRotationEnabled, bugs 327664 /
// 328479) and every upload path (importExternalTexture,
// copyExternalImageToTexture, texImage2D, createImageBitmap) ignores it. The
// <video> element, however, displays the track upright, and its
// videoWidth/videoHeight are the upright size.
//
// So: draw the <video> (upright) and the VideoFrame (sensor orientation) into
// two tiny canvases and pick the rotation/flip that maps one onto the other.
// The dimensions decide the axis (transposed → 90/270, else 0/180); pixels
// decide the direction and mirroring. Runs on the first frame, whenever either
// size changes (the phone turned), and once a second otherwise (landscape-left
// ↔ landscape-right is a 180° change with no size change). ~2 × 32×32 draws +
// readbacks per measurement; nothing per frame.
//
// Engines that expose VideoFrame.rotation (Chrome 138+) apply it themselves in
// every GPU upload path, so this is skipped there (no double rotation). When
// Safari ships the attribute, the same check turns this off automatically.

import { IDENTITY_ORIENTATION, orientationUvMatrix, applyUvMatrix, sameOrientation, type Orientation, type Rotation } from '~/model/orientation.ts'

const GRID = 32                 // long side of the comparison grids
const PERIOD_MS = 1000          // re-measure cadence when sizes are unchanged
const KEEP_RATIO = 0.8          // switch only when the best is clearly better
const MIN_STD = 2 / 255         // flatter than this → inconclusive

export function engineExposesRotation(): boolean {
  return typeof VideoFrame !== 'undefined' && 'rotation' in VideoFrame.prototype
}

export interface LumaGrid { w: number; h: number; data: Float32Array }

// Luma, normalised to zero mean / unit std (robust to exposure / colour
// differences between the two draw paths). null if the grid is ~flat.
export function normalizedLuma(rgba: ArrayLike<number>, w: number, h: number): LumaGrid | null {
  const n = w * h, data = new Float32Array(n)
  let mean = 0
  for (let i = 0; i < n; i++) {
    const v = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255
    data[i] = v; mean += v
  }
  mean /= n
  let varSum = 0
  for (let i = 0; i < n; i++) varSum += (data[i] - mean) ** 2
  const std = Math.sqrt(varSum / n)
  if (std < MIN_STD) return null
  for (let i = 0; i < n; i++) data[i] = (data[i] - mean) / std
  return { w, h, data }
}

export function candidateOrientations(transposed: boolean): Orientation[] {
  const rots: Rotation[] = transposed ? [90, 270] : [0, 180]
  return rots.flatMap(rotation => [{ rotation, flip: false }, { rotation, flip: true }])
}

// Mean |upright − source∘o| over the upright grid (nearest sampling).
export function orientationError(upright: LumaGrid, source: LumaGrid, o: Orientation): number {
  const m = orientationUvMatrix(o)
  let err = 0
  for (let y = 0; y < upright.h; y++) for (let x = 0; x < upright.w; x++) {
    const [su, sv] = applyUvMatrix(m, (x + 0.5) / upright.w, (y + 0.5) / upright.h)
    const sx = Math.min(source.w - 1, Math.max(0, Math.floor(su * source.w)))
    const sy = Math.min(source.h - 1, Math.max(0, Math.floor(sv * source.h)))
    err += Math.abs(upright.data[y * upright.w + x] - source.data[sy * source.w + sx])
  }
  return err / (upright.w * upright.h)
}

// Pick the orientation that maps `source` onto `upright`. Keeps `current` if
// it's a valid candidate and the best isn't clearly better (no flapping on
// near-symmetric scenes, e.g. a centred face and the mirror question).
export function chooseOrientation(
  upright: LumaGrid, source: LumaGrid, transposed: boolean, current: Orientation,
): { orientation: Orientation; errors: Array<{ o: Orientation; err: number }> } {
  const errors = candidateOrientations(transposed).map(o => ({ o, err: orientationError(upright, source, o) }))
  const best = errors.reduce((a, b) => (b.err < a.err ? b : a))
  const cur = errors.find(e => sameOrientation(e.o, current))
  if (cur && !(best.err < cur.err * KEEP_RATIO)) return { orientation: cur.o, errors }
  return { orientation: best.o, errors }
}

const gridSize = (w: number, h: number) => w >= h
  ? { w: GRID, h: Math.max(1, Math.round(GRID * h / w)) }
  : { w: Math.max(1, Math.round(GRID * w / h)), h: GRID }

export interface OrientationProbe {
  // The orientation to apply to `frame` (captured from `video` this tick).
  measure(frame: VideoFrame): Orientation
}

export function createOrientationProbe(
  video: HTMLVideoElement,
  log?: (...a: unknown[]) => void,
): OrientationProbe {
  let current: Orientation = IDENTITY_ORIENTATION
  let lastAt = -Infinity
  let lastKey = ''
  let ctxU: OffscreenCanvasRenderingContext2D | null = null
  let ctxS: OffscreenCanvasRenderingContext2D | null = null
  let broken = false

  const read = (ctx: OffscreenCanvasRenderingContext2D, src: CanvasImageSource, w: number, h: number) => {
    ctx.canvas.width = w; ctx.canvas.height = h
    ctx.drawImage(src, 0, 0, w, h)
    return normalizedLuma(ctx.getImageData(0, 0, w, h).data, w, h)
  }

  return {
    measure(frame) {
      if (broken) return current
      const vw = video.videoWidth, vh = video.videoHeight
      const fw = frame.displayWidth, fh = frame.displayHeight
      if (!vw || !vh || !fw || !fh) return current
      const key = `${vw}x${vh}/${fw}x${fh}`
      const now = performance.now()
      if (key === lastKey && now - lastAt < PERIOD_MS) return current
      lastKey = key
      lastAt = now
      // Square-ish frames can't tell the axis from sizes: treat as not transposed
      // unless the aspect orientations clearly disagree.
      const transposed = (vw > vh) !== (fw > fh) && vw !== vh && fw !== fh
      try {
        ctxU ??= new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D
        ctxS ??= new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D
        const gu = gridSize(vw, vh), gs = gridSize(fw, fh)
        const upright = read(ctxU, video, gu.w, gu.h)
        const source  = read(ctxS, frame, gs.w, gs.h)
        if (!upright || !source) {
          // Flat scene (dark room, lens covered): keep going; if the sizes say
          // the axis changed, at least get the axis right.
          if (transposed !== (current.rotation === 90 || current.rotation === 270)) {
            current = { rotation: transposed ? 90 : 0, flip: current.flip }
            log?.('orientation (inconclusive, axis from sizes):', current)
          }
          lastAt = now - PERIOD_MS + 200   // retry soon
          return current
        }
        const r = chooseOrientation(upright, source, transposed, current)
        if (!sameOrientation(r.orientation, current)) {
          log?.('orientation', current, '→', r.orientation, 'video', vw, vh, 'frame', fw, fh,
            r.errors.map(e => `${e.o.rotation}${e.o.flip ? 'f' : ''}:${e.err.toFixed(3)}`).join(' '))
          current = r.orientation
        }
      } catch (err) {
        // A canvas that can't be drawn/read (tainting, OOM): stop measuring.
        broken = true
        log?.('orientation probe disabled:', err)
      }
      return current
    },
  }
}
