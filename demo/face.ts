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
const JITTER_N = 30   // frames of keypoint history for the jitter (std) meter

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

// Windowed soft-argmax centroid over channel k of the 8-ch NHWC heatmap buffer.
// win=0 degrades to hard argmax (the jitter-demo toggle). Returns continuous
// CELL coords + the peak score.
function decodeKp(hm: Float32Array, h: number, w: number, k: number, win: number) {
  let peak = -Infinity, pr = 0, pc = 0
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = hm[(r * w + c) * 8 + k]
    if (v > peak) { peak = v; pr = r; pc = c }
  }
  if (win === 0) return { x: pc, y: pr, score: peak }
  const r0 = Math.max(0, pr - win), r1 = Math.min(h, pr + win + 1)
  const c0 = Math.max(0, pc - win), c1 = Math.min(w, pc + win + 1)
  let wsum = 0, ry = 0, cx = 0
  for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
    const v = hm[(r * w + c) * 8 + k]
    wsum += v; ry += v * r; cx += v * c
  }
  wsum = Math.max(wsum, 1e-6)
  return { x: cx / wsum, y: ry / wsum, score: peak }
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
      const kps = Array.from({ length: N_KP }, (_, k) => decodeKp(hm, fh, fw, k, win))

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

      // Keypoints (frame-fraction coords: (cell + 0.5) / grid) + jitter history.
      const frameKps: Array<{ x: number; y: number } | null> = []
      const found: number[] = []
      for (let k = 0; k < N_KP; k++) {
        const { x, y, score } = kps[k]
        if (score < thresh) { frameKps.push(null); continue }
        const px = ((x + 0.5) / fw) * view.width
        const py = ((y + 0.5) / fh) * view.height
        frameKps.push({ x: px, y: py })
        found.push(k)
        vctx.strokeStyle = KP_COLORS[k]; vctx.lineWidth = 2
        vctx.beginPath(); vctx.arc(px, py, 5, 0, Math.PI * 2); vctx.stroke()
        vctx.fillStyle = KP_COLORS[k]; vctx.font = '11px ui-monospace'
        vctx.fillText(`${KP_NAMES[k]} ${kps[k].score.toFixed(2)}`, px + 8, py - 6)
      }
      history.push(frameKps)
      if (history.length > JITTER_N) history.shift()

      // Landmark-stage crop preview: square, boxScale × hull long side.
      if (on('showbox') && found.length >= 2) {
        const xs = found.map(k => frameKps[k]!.x), ys = found.map(k => frameKps[k]!.y)
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2
        const cy = (Math.min(...ys) + Math.max(...ys)) / 2
        const side = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * numv('boxscale')
        vctx.strokeStyle = '#fff'; vctx.lineWidth = 1.5; vctx.setLineDash([6, 4])
        vctx.strokeRect(cx - side, cy - side, side * 2, side * 2)
        vctx.setLineDash([])
      }

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
      const scores = kps.map((p, k) => `${KP_NAMES[k]} ${p.score.toFixed(2)}`).join('  ')
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · ${on('soft') ? 'soft' : 'HARD'}-argmax · `
        + `jitter ${Number.isNaN(jitter) ? '—' : jitter.toFixed(2) + 'px'}\n${scores}`)
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['thresh', 'overlay', 'boxscale']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()

$('run').addEventListener('click', () => run())
