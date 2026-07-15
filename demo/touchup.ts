import type { Backend, Dtype, FaceTopology, FaceTouchupStyle, Op, Presenter } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import { LandmarkNet } from '~/model/networks/landmark_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Face touch-up demo — the full flow (see touchup.html). Chain identical to
// landmarks.ts up to the LandmarkNet output; the FaceTouchup presenter then
// consumes the landmark + box tensors as mesh vertex data (no readback).
// Sliders rebuild only what binds them: the box op copies into a stable
// carrier; the touch-up presenter is cheap to rebuild (5 small pipelines).

const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406]
const IMAGENET_STD:  [number, number, number] = [0.229, 0.224, 0.225]
const CROP = 256
const THRESH = 0.15

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

async function loadTopology(): Promise<FaceTopology> {
  const [topoJson, maskBlob] = await Promise.all([
    fetch('/face_topology.json').then(r => r.json()),
    fetch('/weight_mask.png').then(r => r.blob()),
  ])
  return {
    count: topoJson.count,
    uv: new Float32Array(topoJson.uv),
    idx: new Float32Array(topoJson.idx),
    weightMask: await createImageBitmap(maskBlob),
  }
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

    // Fresh canvas per run (a canvas is bound to its first context type).
    const old = $<HTMLCanvasElement>('view')
    const view = document.createElement('canvas')
    view.id = 'view'
    old.replaceWith(view)
    view.width = c.w; view.height = c.h

    status(`loading ${tier} + landmark model + topology…`)
    const [tierBuf, lmBuf, topo, source] = await Promise.all([
      fetchWeights(`/model_${tier}_flow`, dtype),
      fetchWeights('/model_landmark_mesh', dtype),
      loadTopology(),
      loadSource($<HTMLSelectElement>('source').value),
    ])
    stopSource = source.stop
    const backend = await createBackend(backendName, dtype, view)
    const w = loadWeightsFromBinary(tierBuf) as any
    if (!w.face) throw new Error('tier weights have no face blob')
    const lw = loadWeightsFromBinary(lmBuf) as any

    // ── the chain (identical to landmarks.ts up to lm.output) ──────────────
    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const face = new FaceHeatmapNet(backend, matting.encoderTaps, w.face)

    const boxStable = backend.tensor(1, 1, 4, new Float32Array(4))
    let boxOp: Op
    const buildBox = () => {
      boxOp = backend.ops.FaceBoxFromHeatmaps(face.output, {
        win: 3, thresh: THRESH, boxScale: numv('boxscale'),
      })
    }
    buildBox()
    $('boxscale').addEventListener('input', buildBox)

    const crop = backend.ops.CropResample(netInput.output, boxStable, {
      outH: CROP, outW: CROP, mean: IMAGENET_MEAN, std: IMAGENET_STD,
    })
    const lm = new LandmarkNet(backend, crop.output, lw)

    // ── the touch-up presenter (compare = strength 0 → exact passthrough) ──
    let touchup: Presenter
    const buildTouchup = () => {
      touchup = backend.presenters.FaceTouchup(netInput.output, lm.output, boxStable, topo, {
        strength: $<HTMLInputElement>('compare').checked ? 0 : numv('strength'),
        amount: numv('amount'),
        detail: numv('detail'),
        thresh: THRESH,
        style: $<HTMLSelectElement>('style').value as FaceTouchupStyle,
      })
    }
    buildTouchup()
    for (const id of ['strength', 'amount', 'detail']) $(id).addEventListener('input', buildTouchup)
    $('style').addEventListener('change', buildTouchup)
    $('compare').addEventListener('change', buildTouchup)

    let emaFps = 0, last = performance.now()
    status(`running ${tier} — unwrap → freq-sep → composite, GPU-resident`)
    const tick = async () => {
      if (session !== mySession) return
      let bm: ImageBitmap
      try { bm = await source.grab() } catch { return }
      if (session !== mySession) { bm.close(); return }

      netInput.setSource(bm)
      netInput.run()
      matting.run()
      face.run()
      boxOp.run()
      backend.copyTensor(boxOp.output, boxStable)
      crop.run()
      lm.run()
      touchup.run()
      bm.close()

      const now = performance.now()
      emaFps = emaFps ? emaFps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last)
      last = now
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · `
        + `${$<HTMLInputElement>('compare').checked ? 'COMPARE (off)' : 'touch-up on'}`)
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['strength', 'amount', 'detail', 'boxscale']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()

$('run').addEventListener('click', () => run())
