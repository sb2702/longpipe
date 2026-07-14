import type { Backend, Dtype, Tensor, Op } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import { LandmarkNet } from '~/model/networks/landmark_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Landmark probe — the full GPU-resident chain (see landmarks.html). The only
// CPU touch is the optional 16-byte box-tensor readback for the stats line,
// throttled to every 30 frames; untick 'stats readback' for a loop with zero
// readback. Sliders rebuild only the ops that bind them: the box op's output
// is copied into a STABLE box tensor (copyTensor, flowviz's carrier pattern),
// so CropResample / LandmarkNet / the overlay never rebind.

const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406]
const IMAGENET_STD:  [number, number, number] = [0.229, 0.224, 0.225]
const CROP = 256
const N_PTS = 478

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s }
const numv = (id: string) => parseFloat($<HTMLInputElement>(id).value)

async function createBackend(name: string, dtype: Dtype, canvas: HTMLCanvasElement): Promise<Backend> {
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
async function fetchWeights(base: string, dtype: Dtype): Promise<ArrayBuffer> {
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

let session = 0
let stopSource: (() => void) | null = null

async function run() {
  const mySession = ++session
  await new Promise(r => requestAnimationFrame(r))
  stopSource?.(); stopSource = null
  try {
    const tier = $<HTMLSelectElement>('tier').value
    const backendName = $<HTMLSelectElement>('backend').value
    const dtype = $<HTMLSelectElement>('dtype').value as Dtype
    const cfg = TIER_CONFIG[tier]
    const c = cfg.canvasRes

    // Fresh canvas per run — a canvas is permanently bound to its first
    // context type, so switching webgpu ↔ webgl needs a new element.
    const old = $<HTMLCanvasElement>('view')
    const view = document.createElement('canvas')
    view.id = 'view'
    old.replaceWith(view)
    view.width = c.w; view.height = c.h

    status(`loading ${tier} + landmark model…`)
    const [tierBuf, lmBuf, source] = await Promise.all([
      fetchWeights(`/model_${tier}_flow`, dtype),
      fetchWeights('/model_landmark_mesh', dtype),
      loadSource($<HTMLSelectElement>('source').value),
    ])
    stopSource = source.stop
    const backend = await createBackend(backendName, dtype, view)
    const w = loadWeightsFromBinary(tierBuf) as any
    if (!w.face) throw new Error('tier weights have no face blob')
    const lw = loadWeightsFromBinary(lmBuf) as any

    // ── the chain ──────────────────────────────────────────────────────────
    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const face = new FaceHeatmapNet(backend, matting.encoderTaps, w.face)

    // Stable box carrier: sliders rebuild only boxOp; consumers bind this.
    const boxStable = backend.tensor(1, 1, 4, new Float32Array(4))
    let boxOp: Op
    const buildBox = () => {
      boxOp = backend.ops.FaceBoxFromHeatmaps(face.output, {
        win: 3, thresh: numv('thresh'), boxScale: numv('boxscale'),
      })
    }
    buildBox()
    $('boxscale').addEventListener('input', buildBox)
    $('thresh').addEventListener('input', buildBox)

    const crop = backend.ops.CropResample(netInput.output, boxStable, {
      outH: CROP, outW: CROP, mean: IMAGENET_MEAN, std: IMAGENET_STD,
    })
    const lm = new LandmarkNet(backend, crop.output, lw)

    let overlay = backend.presenters.LandmarkOverlay(netInput.output, lm.output, boxStable, {
      count: N_PTS, thresh: Math.max(numv('thresh'), 1e-3), pointSize: numv('ptsize'), color: [0.06, 0.72, 0.5],
    })
    const buildOverlay = () => {
      overlay = backend.presenters.LandmarkOverlay(netInput.output, lm.output, boxStable, {
        count: N_PTS, thresh: Math.max(numv('thresh'), 1e-3), pointSize: numv('ptsize'), color: [0.06, 0.72, 0.5],
      })
    }
    $('ptsize').addEventListener('input', buildOverlay)
    $('thresh').addEventListener('input', buildOverlay)

    let emaFps = 0, last = performance.now(), frame = 0, lastScore = '—'
    status(`running ${tier} — crop ${CROP}², ${N_PTS} pts, GPU-resident`)
    const tick = async () => {
      if (session !== mySession) return
      let bm: ImageBitmap
      try { bm = await source.grab() } catch { return }
      if (session !== mySession) { bm.close(); return }

      netInput.setSource(bm)
      netInput.run()
      matting.run()                              // encoder taps (production shape)
      face.run()
      boxOp.run()
      backend.copyTensor(boxOp.output, boxStable)
      crop.run()
      lm.run()
      overlay.run()
      bm.close()

      frame++
      if ($<HTMLInputElement>('stats').checked && frame % 30 === 0) {
        const b = await backend.readback(boxStable)   // 16 bytes, every 30 frames
        lastScore = `score ${b[3].toFixed(2)} · box (${b[0].toFixed(2)}, ${b[1].toFixed(2)}) hs ${b[2].toFixed(3)}`
      }
      const now = performance.now()
      emaFps = emaFps ? emaFps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last)
      last = now
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · ${lastScore}`)
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['boxscale', 'thresh', 'ptsize']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()

$('run').addEventListener('click', () => run())
