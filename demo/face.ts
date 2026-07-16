import type { Backend, Dtype } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Face-heatmap probe — the production face path in isolation: TierModel matting
// pass (which computes the encoder taps) → FaceHeatmapNet → 5 keypoint heatmaps
// at base/4. Decode is the WINDOWED SOFT-ARGMAX centroid (±3 cells around the
// peak) — the contract from training/eval/visualize_live_webcam.py; the 'soft'
// toggle switches to hard argmax to demonstrate the cell-snapping jitter it
// prevents. The dashed square previews the crop the landmark stage would take
// (square, boxScale × the keypoint hull's long side, centered on the hull).
//
// Heatmap readback here is ~10-40 KB/frame — fine for a probe page; the
// production landmark path keeps the decode + crop on the GPU.

const WIN = 3
const N_KP = 5
const KP_NAMES = ['L-eye', 'R-eye', 'nose', 'L-mouth', 'R-mouth']
const KP_COLORS = ['#ff5f5f', '#54e08c', '#ffd23f', '#5fb8ff', '#e05fd8']
const FACE_COLORS = ['#ffffff', '#ffd23f', '#54e08c', '#5fb8ff', '#e05fd8', '#ff8a3d']
const JITTER_N = 30   // frames of keypoint history for the jitter (std) meter
const MIN_SEP = 1.0   // cells — two refined centroids closer than this are one blob
const LAMBDA  = 0.5   // geometric-residual weight in the hypothesis score

// ArcFace / RetinaFace canonical 5-point template (112² reference frame), in the
// head's channel order: L-eye, R-eye, nose, L-mouth, R-mouth.
const CANON: [number, number][] = [
  [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
  [41.5493, 92.3655], [70.7299, 92.2041],
]

// The template re-expressed in the EYE FRAME: origin = L-eye, +u along the
// L-eye→R-eye vector, +w perpendicular to it (image-down for an upright face),
// both normalized by interocular distance. An eye-pair hypothesis fixes origin,
// scale and roll, so predicting the other three points is just (u, w) × that
// frame. It's a similarity with no reflection term — which is why a FLIPPED eye
// convention (if the head's channel 0 is actually the image-right eye) shows up
// as "no faces detected" rather than as silently mirrored boxes: the predicted
// nose/mouth land above the eyes and match nothing.
const TEMPLATE = (() => {
  const [e0, e1] = [CANON[0], CANON[1]]
  const ev = [e1[0] - e0[0], e1[1] - e0[1]]
  const elen = Math.hypot(ev[0], ev[1])
  const ex = [ev[0] / elen, ev[1] / elen]
  const ey = [-ex[1], ex[0]]
  return CANON.map(p => {
    const v = [p[0] - e0[0], p[1] - e0[1]]
    return {
      u: (v[0] * ex[0] + v[1] * ex[1]) / elen,
      w: (v[0] * ey[0] + v[1] * ey[1]) / elen,
    }
  })
})()

interface Cand { x: number; y: number; score: number }
interface Face { kp: (Cand | null)[]; keys: string[]; score: number }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s }
const numv = (id: string) => parseFloat($<HTMLInputElement>(id).value)
const on = (id: string) => $<HTMLInputElement>(id).checked

async function createBackend(name: string, dtype: Dtype, canvas: HTMLCanvasElement | OffscreenCanvas): Promise<Backend> {
  if (name === 'webgpu') {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    if (dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) throw new Error('adapter lacks shader-f16')
    return WebGPUBackend.create({ canvas, dtype })
  }
  return WebGLBackend.create({ canvas, dtype })
}

async function fetchBin(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url)
  if (!r.ok || (r.headers.get('content-type') ?? '').startsWith('text/html')) return null
  return r.arrayBuffer()
}
async function fetchWeights(tier: string, dtype: Dtype): Promise<ArrayBuffer> {
  const base = `/model_${tier}_flow`   // the current small/xs exports (flow + face)
  if (dtype === 'f16') { const f16 = await fetchBin(`${base}.f16.bin`); if (f16) return f16 }
  const f32 = await fetchBin(`${base}.bin`)
  if (!f32) throw new Error(`failed to fetch ${base}.bin`)
  return f32
}

type Source = { grab: () => Promise<ImageBitmap>; stop: () => void }

async function loadSource(kind: string): Promise<Source> {
  if (kind === 'image') {
    const blob = await (await fetch('/test_img.jpg')).blob()
    const bm = await createImageBitmap(blob)
    return { grab: () => createImageBitmap(bm), stop: () => bm.close() }
  }
  const v = document.createElement('video')
  v.muted = true; v.playsInline = true
  if (kind === 'video') {
    v.src = '/loop_video.mp4'; v.loop = true
  } else {
    v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
  }
  await v.play()
  return {
    grab: () => createImageBitmap(v),
    stop: () => {
      v.pause()
      if (v.srcObject) for (const t of (v.srcObject as MediaStream).getTracks()) t.stop()
    },
  }
}

// Windowed soft-argmax centroid around (pr, pc) on channel k. Returns continuous
// CELL coords. NOTE for multi-face: the ±win window is blind to face identity —
// two faces closer than `win` cells drag each other's centroids. That's a real
// limit of this decode on coarse grids, not a bug to patch here.
function centroid(hm: Float32Array, h: number, w: number, k: number, pr: number, pc: number, win: number) {
  const r0 = Math.max(0, pr - win), r1 = Math.min(h, pr + win + 1)
  const c0 = Math.max(0, pc - win), c1 = Math.min(w, pc + win + 1)
  let wsum = 0, ry = 0, cx = 0
  for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
    const v = hm[(r * w + c) * 8 + k]
    wsum += v; ry += v * r; cx += v * c
  }
  wsum = Math.max(wsum, 1e-6)
  return { x: cx / wsum, y: ry / wsum }
}

// SINGLE-FACE decode — the current production contract (face_box.wgsl): global
// argmax per channel + windowed centroid. win=0 degrades to hard argmax (the
// jitter-demo toggle). Kept so the 'multi-face' toggle is a live A/B.
function decodeKp(hm: Float32Array, h: number, w: number, k: number, win: number): Cand {
  let peak = -Infinity, pr = 0, pc = 0
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = hm[(r * w + c) * 8 + k]
    if (v > peak) { peak = v; pr = r; pc = c }
  }
  if (win === 0) return { x: pc, y: pr, score: peak }
  return { ...centroid(hm, h, w, k, pr, pc, win), score: peak }
}

// STAGE 1 — local-maximum candidates for channel k. A cell qualifies if it's the
// max of its 3×3 neighborhood (index tiebreak so f16 plateaus don't drop every
// member of a tie) and above thresh. Sorted by peak score, capped at maxK, then
// deduped by refined position — two local maxima on one noisy blob converge to
// nearly the same centroid.
function findCandidates(hm: Float32Array, h: number, w: number, k: number,
                        win: number, thresh: number, maxK: number): Cand[] {
  const at = (r: number, c: number) => hm[(r * w + c) * 8 + k]
  const peaks: { r: number; c: number; score: number }[] = []
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = at(r, c)
    if (v < thresh) continue
    let isMax = true
    for (let dr = -1; dr <= 1 && isMax; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue
      const rr = r + dr, cc = c + dc
      if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue
      const u = at(rr, cc)
      if (u > v || (u === v && rr * w + cc < r * w + c)) { isMax = false; break }
    }
    if (isMax) peaks.push({ r, c, score: v })
  }
  peaks.sort((a, b) => b.score - a.score)

  const out: Cand[] = []
  for (const p of peaks) {
    if (out.length >= maxK) break
    const cand: Cand = win === 0
      ? { x: p.c, y: p.r, score: p.score }
      : { ...centroid(hm, h, w, k, p.r, p.c, win), score: p.score }
    if (out.some(o => Math.hypot(o.x - cand.x, o.y - cand.y) < MIN_SEP)) continue
    out.push(cand)
  }
  return out
}

// STAGE 2/3/4 — eye-pair hypotheses → geometric scoring → face-level NMS.
//
// Every (L-eye, R-eye) candidate pair is a face hypothesis. The pair fixes the
// face's center, scale (interocular distance — the classic normalizer) and roll,
// so nose + mouth corners get PREDICTED from the canonical template and matched
// against nearby candidates within tol × interocular. Cross-pairing (face A's
// left eye with face B's right eye) survives stage 1 but dies here: the implied
// scale is wrong and there's no nose/mouth support where the template says.
//
// Scoring = mean matched peak score − λ · mean normalized residual. Then greedy
// NMS: accept hypotheses by score, rejecting any that reuse a consumed candidate.
function groupFaces(cands: Cand[][], opts: { tol: number; maxFaces: number; gridW: number }): Face[] {
  const MIN_EYE = 1.0                  // cells — below this the grid can't resolve a pair
  const MAX_EYE = opts.gridW * 0.45    // a face wider than ~half the frame
  const hyps: Face[] = []

  for (let i = 0; i < cands[0].length; i++) for (let j = 0; j < cands[1].length; j++) {
    const a = cands[0][i], b = cands[1][j]
    const evx = b.x - a.x, evy = b.y - a.y
    const len = Math.hypot(evx, evy)
    if (len < MIN_EYE || len > MAX_EYE) continue
    const exv = [evx / len, evy / len]
    const eyv = [-exv[1], exv[0]]

    const kp: (Cand | null)[] = [a, b, null, null, null]
    const keys = [`0:${i}`, `1:${j}`]
    let sc = a.score + b.score, resid = 0, nMatch = 2, mouths = 0

    for (let k = 2; k < N_KP; k++) {
      const t = TEMPLATE[k]
      const px = a.x + len * (t.u * exv[0] + t.w * eyv[0])
      const py = a.y + len * (t.u * exv[1] + t.w * eyv[1])
      let best = -1, bestD = opts.tol * len
      for (let m = 0; m < cands[k].length; m++) {
        const d = Math.hypot(cands[k][m].x - px, cands[k][m].y - py)
        if (d < bestD) { bestD = d; best = m }
      }
      if (best < 0) continue
      kp[k] = cands[k][best]
      keys.push(`${k}:${best}`)
      sc += cands[k][best].score
      resid += bestD / len
      nMatch++
      if (k >= 3) mouths++
    }
    if (!kp[2] || mouths < 1) continue   // support floor: nose + ≥1 mouth corner
    hyps.push({ kp, keys, score: sc / nMatch - LAMBDA * (resid / (nMatch - 2)) })
  }

  hyps.sort((x, y) => y.score - x.score)
  const used = new Set<string>()
  const out: Face[] = []
  for (const hy of hyps) {
    if (out.length >= opts.maxFaces) break
    if (hy.keys.some(k => used.has(k))) continue
    for (const k of hy.keys) used.add(k)
    out.push(hy)
  }
  return out
}

let session = 0
let stopSource: (() => void) | null = null

async function run() {
  // New session invalidates any in-flight tick BEFORE the old source is torn
  // down (an in-flight grab on a closed source throws otherwise).
  const mySession = ++session
  await new Promise(r => requestAnimationFrame(r))
  stopSource?.(); stopSource = null
  try {
    const tier = $<HTMLSelectElement>('tier').value
    const backendName = $<HTMLSelectElement>('backend').value
    const dtype = $<HTMLSelectElement>('dtype').value as Dtype
    const cfg = TIER_CONFIG[tier]
    const c = cfg.canvasRes

    status(`loading ${tier}…`)
    const [wbuf, source] = await Promise.all([
      fetchWeights(tier, dtype),
      loadSource($<HTMLSelectElement>('source').value),
    ])
    stopSource = source.stop
    const backend = await createBackend(backendName, dtype, new OffscreenCanvas(4, 4))
    const w = loadWeightsFromBinary(wbuf) as any
    if (!w.face) throw new Error('weights have no face blob — re-export with a face-trained checkpoint')

    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const face = new FaceHeatmapNet(backend, matting.encoderTaps, w.face)
    const fh = face.output.h, fw = face.output.w

    // Display canvases. View at canvas res × integer scale for crisp mapping.
    const view = $<HTMLCanvasElement>('view'); view.width = c.w * 2; view.height = c.h * 2
    const vctx = view.getContext('2d')!
    const raw = $<HTMLCanvasElement>('raw'); raw.width = fw; raw.height = fh
    const rctx = raw.getContext('2d')!
    const rawImg = rctx.createImageData(fw, fh)
    const hmCanvas = new OffscreenCanvas(fw, fh)
    const hmCtx = hmCanvas.getContext('2d')!

    const history: Array<Array<{ x: number; y: number } | null>> = []
    let emaFps = 0, last = performance.now()

    status(`running ${tier} — face grid ${fw}×${fh} (base/4)`)
    const tick = async () => {
      if (session !== mySession) return
      let bm: ImageBitmap
      try { bm = await source.grab() } catch { return }
      if (session !== mySession) { bm.close(); return }
      netInput.setSource(bm)
      netInput.run()
      matting.run()          // computes encoderTaps as a side effect (production shape)
      face.run()
      const hm = await backend.readback(face.output)

      const win = on('soft') ? WIN : 0
      const thresh = numv('thresh')
      const multi = on('multi')
      const maxFaces = Math.round(numv('maxfaces'))

      // Decode. multi: local-max candidates → eye-pair grouping → K faces.
      // single: the production global-argmax path, wrapped as one face.
      let faces: Face[] = []
      let candsAll: Cand[][] = []
      if (multi) {
        // maxFaces+1 candidates/channel — headroom so cross-pairs exist to reject.
        candsAll = Array.from({ length: N_KP },
          (_, k) => findCandidates(hm, fh, fw, k, win, thresh, maxFaces + 1))
        faces = groupFaces(candsAll, { tol: numv('tol'), maxFaces, gridW: fw })
      } else {
        const kps = Array.from({ length: N_KP }, (_, k) => decodeKp(hm, fh, fw, k, win))
        const kp = kps.map(p => (p.score >= thresh ? p : null))
        if (kp.filter(Boolean).length >= 2) {
          faces = [{ kp, keys: [], score: Math.min(...kps.map(p => p.score)) }]
        }
      }

      // ── main view: frame + heatmap tint + keypoints + crop box ────────────
      vctx.drawImage(bm, 0, 0, view.width, view.height)
      bm.close()

      // Raw heatmap panel always updates; the main-view overlay is gated below.
      {
        for (let p = 0; p < fh * fw; p++) {
          let best = 0, bk = 0
          for (let k = 0; k < N_KP; k++) {
            const v = hm[p * 8 + k]
            if (v > best) { best = v; bk = k }
          }
          const col = KP_COLORS[bk]
          rawImg.data[p * 4]     = parseInt(col.slice(1, 3), 16)
          rawImg.data[p * 4 + 1] = parseInt(col.slice(3, 5), 16)
          rawImg.data[p * 4 + 2] = parseInt(col.slice(5, 7), 16)
          rawImg.data[p * 4 + 3] = Math.min(255, best * 255 * 1.5)
        }
        rctx.fillStyle = '#000'; rctx.fillRect(0, 0, fw, fh)
        rctx.putImageData(rawImg, 0, 0)
      }
      if (on('showhm')) {
        hmCtx.putImageData(rawImg, 0, 0)
        vctx.save()
        vctx.globalAlpha = numv('overlay')
        vctx.imageSmoothingEnabled = true
        vctx.drawImage(hmCanvas, 0, 0, view.width, view.height)
        vctx.restore()
      }

      // Cell coords → view px. (cell + 0.5) / grid = frame fraction.
      const toView = (c: Cand) => ({
        x: ((c.x + 0.5) / fw) * view.width,
        y: ((c.y + 0.5) / fh) * view.height,
      })

      // Raw candidates (dim) — shows what stage 1 found before grouping, so a
      // dropped face reads as "no candidate" vs "grouping rejected it".
      if (multi && on('cands')) {
        vctx.globalAlpha = 0.35
        for (let k = 0; k < N_KP; k++) for (const c of candsAll[k]) {
          const p = toView(c)
          vctx.fillStyle = KP_COLORS[k]
          vctx.beginPath(); vctx.arc(p.x, p.y, 3, 0, Math.PI * 2); vctx.fill()
        }
        vctx.globalAlpha = 1
      }

      // Per-face: keypoints in CHANNEL colors, box + label in FACE color.
      faces.forEach((f, fi) => {
        const col = FACE_COLORS[fi % FACE_COLORS.length]
        const pts: Array<{ x: number; y: number }> = []
        for (let k = 0; k < N_KP; k++) {
          const c = f.kp[k]
          if (!c) continue
          const p = toView(c)
          pts.push(p)
          vctx.strokeStyle = KP_COLORS[k]; vctx.lineWidth = 2
          vctx.beginPath(); vctx.arc(p.x, p.y, 5, 0, Math.PI * 2); vctx.stroke()
          if (!multi) {   // single-face mode keeps the per-keypoint labels
            vctx.fillStyle = KP_COLORS[k]; vctx.font = '11px ui-monospace'
            vctx.fillText(`${KP_NAMES[k]} ${c.score.toFixed(2)}`, p.x + 8, p.y - 6)
          }
        }
        // Landmark-stage crop preview: square, boxScale × hull long side.
        if (on('showbox') && pts.length >= 2) {
          const xs = pts.map(p => p.x), ys = pts.map(p => p.y)
          const cx = (Math.min(...xs) + Math.max(...xs)) / 2
          const cy = (Math.min(...ys) + Math.max(...ys)) / 2
          const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * numv('boxscale')
          vctx.strokeStyle = col; vctx.lineWidth = 1.5; vctx.setLineDash([6, 4])
          vctx.strokeRect(cx - side, cy - side, side * 2, side * 2)
          vctx.setLineDash([])
          if (multi) {
            vctx.fillStyle = col; vctx.font = '12px ui-monospace'
            vctx.fillText(`#${fi} ${f.score.toFixed(2)}`, cx - side + 4, cy - side - 5)
          }
        }
      })

      // Jitter history tracks the TOP-SCORING face only. With two faces of
      // similar score the identity can swap between frames — read the meter as
      // primary-subject stability, not a per-face guarantee.
      history.push(faces[0] ? faces[0].kp.map(c => (c ? toView(c) : null)) : new Array(N_KP).fill(null))
      if (history.length > JITTER_N) history.shift()

      // Jitter meter: mean per-keypoint std over the history window (canvas px).
      let jitter = NaN
      if (history.length >= 10) {
        const stds: number[] = []
        for (let k = 0; k < N_KP; k++) {
          const pts = history.map(f => f[k]).filter(Boolean) as Array<{ x: number; y: number }>
          if (pts.length < history.length * 0.8) continue
          const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length
          const my = pts.reduce((s, p) => s + p.y, 0) / pts.length
          stds.push(Math.sqrt(pts.reduce((s, p) => s + (p.x - mx) ** 2 + (p.y - my) ** 2, 0) / pts.length))
        }
        if (stds.length) jitter = stds.reduce((a, b) => a + b, 0) / stds.length
      }

      const now = performance.now()
      emaFps = emaFps ? emaFps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last)
      last = now
      const detail = multi
        ? `cands ${candsAll.map(c => c.length).join('/')} (${KP_NAMES.join('/')})\n`
          + (faces.length
            ? faces.map((f, i) => `#${i} score ${f.score.toFixed(2)} · ${f.kp.filter(Boolean).length}/5 kp`).join('\n')
            : 'no faces grouped')
        : faces[0]
          ? KP_NAMES.map((n, k) => `${n} ${faces[0].kp[k]?.score.toFixed(2) ?? '—'}`).join('  ')
          : 'no face'
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · ${on('soft') ? 'soft' : 'HARD'}-argmax · `
        + `${multi ? `multi: ${faces.length} face(s)` : 'single'} · `
        + `jitter ${Number.isNaN(jitter) ? '—' : jitter.toFixed(2) + 'px'}\n${detail}`)
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['thresh', 'overlay', 'boxscale', 'maxfaces', 'tol']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()

$('run').addEventListener('click', () => run())
