import { describe, it, expect } from 'vitest'
import { createWebGPUBackend, createWebGLBackend } from '../helpers/backends'
import type { Backend } from '~/model/backend'
import {
  orientationUvMatrix, applyUvMatrix, uprightSize, type Orientation, type Rotation,
} from '~/model/orientation'

// Source: W×H, every pixel a unique colour (R = x, G = y, B = 255 - id), so any
// misplaced sample is visible exactly. Sampling at upright pixel centres maps
// to source pixel centres, so bilinear filtering returns exact texel values.
const W = 4, H = 2
function sourcePixels(): Uint8ClampedArray {
  const px = new Uint8ClampedArray(W * H * 4)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4
    px.set([x * 60 + 10, y * 100 + 20, 255 - (y * W + x) * 20, 255], i)
  }
  return px
}

// Reference: rotate clockwise, THEN flip horizontally (WebCodecs rendering
// order), by explicit pixel moves — independent of orientationUvMatrix.
function referenceUpright(src: Uint8ClampedArray, o: Orientation): { w: number; h: number; px: Uint8ClampedArray } {
  let w = W, h = H, cur = src
  for (let k = 0; k < o.rotation / 90; k++) {
    const nw = h, nh = w, next = new Uint8ClampedArray(cur.length)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      // 90° CW: (x, y) → (h-1-y, x)
      const nx = h - 1 - y, ny = x
      next.set(cur.subarray((y * w + x) * 4, (y * w + x) * 4 + 4), (ny * nw + nx) * 4)
    }
    cur = next; w = nw; h = nh
  }
  if (o.flip) {
    const next = new Uint8ClampedArray(cur.length)
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
      next.set(cur.subarray((y * w + x) * 4, (y * w + x) * 4 + 4), (y * w + (w - 1 - x)) * 4)
    cur = next
  }
  return { w, h, px: cur }
}

const CASES: Orientation[] = ([0, 90, 180, 270] as Rotation[])
  .flatMap(rotation => [{ rotation, flip: false }, { rotation, flip: true }])
const label = (o: Orientation) => `rot ${o.rotation}${o.flip ? ' + flip' : ''}`

describe('orientationUvMatrix (CPU)', () => {
  it.each(CASES.map(o => [label(o), o] as const))('%s matches rotate-then-flip pixel moves', (_l, o) => {
    const src = sourcePixels()
    const ref = referenceUpright(src, o)
    const m = orientationUvMatrix(o)
    const size = uprightSize(W, H, o)
    expect(size).toEqual({ w: ref.w, h: ref.h })
    for (let y = 0; y < ref.h; y++) for (let x = 0; x < ref.w; x++) {
      const [su, sv] = applyUvMatrix(m, (x + 0.5) / ref.w, (y + 0.5) / ref.h)
      const sx = Math.floor(su * W), sy = Math.floor(sv * H)
      expect(Array.from(src.subarray((sy * W + sx) * 4, (sy * W + sx) * 4 + 4)))
        .toEqual(Array.from(ref.px.subarray((y * ref.w + x) * 4, (y * ref.w + x) * 4 + 4)))
    }
  })

  // The browser that implements VideoFrame.rotation/flip is the semantics
  // authority: our reference must match what it renders.
  it('reference matches Chromium rendering of VideoFrame{rotation, flip}', async () => {
    const probe = new VideoFrame(new Uint8Array(4 * 4), { format: 'RGBA', codedWidth: 2, codedHeight: 2, timestamp: 0, rotation: 90 } as VideoFrameBufferInit)
    const supported = (probe as unknown as { rotation?: number }).rotation === 90
    probe.close()
    if (!supported) { console.warn('skip: VideoFrame rotation init unsupported in this browser'); return }
    for (const o of CASES) {
      const frame = new VideoFrame(sourcePixels(), { format: 'RGBA', codedWidth: W, codedHeight: H, timestamp: 0, rotation: o.rotation, flip: o.flip } as VideoFrameBufferInit)
      // drawImage, not createImageBitmap: Chrome 147 (full build) returns an
      // UNROTATED bitmap from createImageBitmap(frame) (headless-shell rotates).
      const ref = referenceUpright(sourcePixels(), o)
      expect([frame.displayWidth, frame.displayHeight], label(o)).toEqual([ref.w, ref.h])
      const c = new OffscreenCanvas(ref.w, ref.h).getContext('2d')!
      c.drawImage(frame, 0, 0)
      frame.close()
      const got = c.getImageData(0, 0, ref.w, ref.h).data
      for (let i = 0; i < got.length; i++) expect(Math.abs(got[i] - ref.px[i]), `${label(o)} byte ${i}`).toBeLessThanOrEqual(2)
    }
  })
})

type Make = () => Promise<Backend>
const BACKENDS: Array<{ name: string; create: Make }> = [
  { name: 'WebGPU', create: () => createWebGPUBackend('f32') },
  { name: 'WebGL',  create: async () => createWebGLBackend('f32') },
]

async function runInput(backend: Backend, src: ImageBitmap | VideoFrame, o: Orientation): Promise<{ w: number; h: number; px: number[] }> {
  const { w, h } = uprightSize(W, H, o)
  const op = backend.ops.Input(h, w)
  op.setSource(src, o)
  op.run()
  const out = await backend.readback(op.output)
  const px: number[] = []
  for (let i = 0; i < w * h * 4; i++) px.push(Math.round(out[i] * 255))
  return { w, h, px }
}

describe.each(BACKENDS)('Input op orientation ($name)', ({ create }) => {
  it.each(CASES.map(o => [label(o), o] as const))('ImageBitmap source, %s', async (_l, o) => {
    const backend = await create()
    try {
      const bmp = await createImageBitmap(new ImageData(sourcePixels(), W, H), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
      const got = await runInput(backend, bmp, o)
      const ref = referenceUpright(sourcePixels(), o)
      for (let i = 0; i < got.px.length; i++) expect(Math.abs(got.px[i] - ref.px[i]), `byte ${i}`).toBeLessThanOrEqual(1)
    } finally { backend.destroy() }
  })

  it.each(CASES.map(o => [label(o), o] as const))('VideoFrame source (external / texImage2D path), %s', async (_l, o) => {
    const backend = await create()
    const frame = new VideoFrame(sourcePixels(), { format: 'RGBA', codedWidth: W, codedHeight: H, timestamp: 0 })
    try {
      const got = await runInput(backend, frame, o)
      const ref = referenceUpright(sourcePixels(), o)
      // Video import may go through a YUV/colour-managed path: allow small drift.
      for (let i = 0; i < got.px.length; i++) expect(Math.abs(got.px[i] - ref.px[i]), `byte ${i}`).toBeLessThanOrEqual(4)
    } finally { frame.close(); backend.destroy() }
  })

  it('orientation changes between runs take effect (uniform rewrite)', async () => {
    const backend = await create()
    try {
      const bmp = await createImageBitmap(new ImageData(sourcePixels(), W, H), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
      const op = backend.ops.Input(W, W)   // square output: shapes don't constrain the result
      const sample = async (o: Orientation) => { op.setSource(bmp, o); op.run(); return Array.from(await backend.readback(op.output)).slice(0, 4).map(v => Math.round(v * 255)) }
      const a = await sample({ rotation: 0, flip: false })
      const b = await sample({ rotation: 180, flip: false })
      const c = await sample({ rotation: 0, flip: false })
      expect(a).toEqual(c)
      expect(a).not.toEqual(b)
    } finally { backend.destroy() }
  })
})
