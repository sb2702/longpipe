import type { Backend, Dtype, Tensor, Op } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Flow-stabilizer probe — the EXACT SDK stabilizer (same ops as the renderer),
// isolated from the production pipeline (no skip cadence, preview, applyAlpha, or
// display upscale). Runs every frame like a skipFrames=0 tier:
//   flow(prev→cur) → matting(pred) → [warp ref] → gate-blend → carrier
// Panels: input | flow (HSV) | STABILIZED greenscreen. Sliders mirror the Python
// harness trackbars so settings are directly comparable.

const DEC_W = 16
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
async function fetchWeights(tier: string, dtype: Dtype): Promise<ArrayBuffer> {
  const base = `/model_${tier}_flow`
  if (dtype === 'f16') { const f16 = await fetchBin(`${base}.f16.bin`); if (f16) return f16 }
  const f32 = await fetchBin(`${base}.bin`)
  if (!f32) throw new Error(`failed to fetch ${base}.bin`)
  return f32
}
async function loadWebcam(): Promise<HTMLVideoElement> {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
  const v = document.createElement('video')
  v.srcObject = stream
  v.muted = true; v.playsInline = true
  await v.play()
  return v
}

// h 0..360, s/v 0..1 → [r,g,b] 0..255
function hsv(h: number, s: number, v: number): [number, number, number] {
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c
  let r = 0, g = 0, b = 0
  if (h < 60) { r = c; g = x } else if (h < 120) { r = x; g = c } else if (h < 180) { g = c; b = x }
  else if (h < 240) { g = x; b = c } else if (h < 300) { r = x; b = c } else { r = c; b = x }
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255]
}

let running = false

async function run() {
  running = false
  await new Promise(r => requestAnimationFrame(r))
  running = true
  try {
    const tier = $<HTMLSelectElement>('tier').value
    const backendName = $<HTMLSelectElement>('backend').value
    const dtype = $<HTMLSelectElement>('dtype').value as Dtype
    const cfg = TIER_CONFIG[tier]
    const c = cfg.canvasRes, b = cfg.baseRes

    const green = $<HTMLCanvasElement>('green'); green.width = c.w; green.height = c.h
    const inputC = $<HTMLCanvasElement>('input'); inputC.width = c.w; inputC.height = c.h
    const inputCtx = inputC.getContext('2d')!
    const flowC = $<HTMLCanvasElement>('flow')

    status(`loading ${tier}…`)
    const [wbuf, video] = await Promise.all([fetchWeights(tier, dtype), loadWebcam()])
    const backend = await createBackend(backendName, dtype, green)
    const w = loadWeightsFromBinary(wbuf) as any
    if (!w.flow) throw new Error('weights have no flow blob')
    const zeros = (h: number, wd: number) => backend.tensor(h, wd, 4, new Float32Array(h * wd * 4))

    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const curBaseDown = backend.ops.BilinearUpsample(netInput.output, { outH: b.h, outW: b.w })
    const frameAHeld = zeros(b.h, b.w)
    const flow = new OpticalFlowNet(backend, frameAHeld, curBaseDown.output, matting.encoderTaps, w.flow, DEC_W,
      tier === 'xs' ? { fuseStem: true, halfTap: matting.halfTap } : {})
    const up = backend.ops.BilinearUpsample(flow.output, { outH: c.h, outW: c.w })
    const flowScale = -(c.w / b.w)
    const stepX = Math.max(1, Math.round(c.w / flow.output.w))
    const stepY = Math.max(1, Math.round(c.h / flow.output.h))

    const stabPrev = zeros(c.h, c.w)                 // (.x stab, .y env) carrier — stable tensor
    const refWarp  = backend.ops.Warp(stabPrev, up.output, { flowScale })
    const greenComp = backend.presenters.CompositeSolid(netInput.output, stabPrev, [0, 1, 0])

    // Rebuild the stabilize op when a slider / warp toggle changes (params + the
    // ref source bind at construction). Its output is copied into stabPrev, which
    // greenComp composites — so greenComp never needs rebinding.
    let stab: Op
    const buildStab = () => {
      const warpOn = $<HTMLInputElement>('warp').checked
      stab = backend.ops.Stabilize(up.output, matting.output, warpOn ? refWarp.output : stabPrev, stabPrev, {
        tLo: numv('tLo'), tHi: numv('tHi'), leak: numv('leak'), release: numv('release'),
        tDiv: numv('tDiv'), divScale: numv('divScale'), stepX, stepY,
      })
    }
    buildStab()
    for (const id of ['tLo', 'tHi', 'leak', 'release', 'tDiv', 'divScale'])
      $(id).addEventListener('input', buildStab)
    $('warp').addEventListener('change', buildStab)

    const fw = flow.output.w, fh = flow.output.h
    flowC.width = fw; flowC.height = fh
    const flowCtx = flowC.getContext('2d')!
    const flowImg = flowCtx.createImageData(fw, fh)

    let warm = false
    status(`running ${tier} — input | flow (base/4 ${fw}×${fh}) | stabilized`)
    const tick = async () => {
      if (!running) return
      const bm = await createImageBitmap(video)
      netInput.setSource(bm)
      netInput.run()
      curBaseDown.run()                              // canvas → base (matches training stem input)
      flow.run(); up.run()                           // flow before matting → taps are prev's
      matting.run()
      backend.copyTensor(curBaseDown.output, frameAHeld)

      if (!warm) {                                   // seed carrier with first matte
        backend.copyTensor(matting.output, stabPrev)
        warm = true
      } else {
        if ($<HTMLInputElement>('warp').checked) refWarp.run()
        stab.run()
        backend.copyTensor(stab.output, stabPrev)
      }
      greenComp.run()

      const fd = await backend.readback(flow.output)
      const gain = numv('gain')
      for (let p = 0; p < fh * fw; p++) {
        const fx = fd[p * 4], fy = fd[p * 4 + 1]
        const mag = Math.hypot(fx, fy)
        const ang = (Math.atan2(fy, fx) + Math.PI) / (2 * Math.PI) * 360
        const [r, g, bl] = hsv(ang, 1, Math.min(1, mag * gain))
        flowImg.data[p * 4] = r; flowImg.data[p * 4 + 1] = g; flowImg.data[p * 4 + 2] = bl; flowImg.data[p * 4 + 3] = 255
      }
      flowCtx.putImageData(flowImg, 0, 0)
      inputCtx.drawImage(bm, 0, 0, inputC.width, inputC.height)
      bm.close()
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    running = false
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

// Live value readouts next to each slider (wired at load — sliders exist before Run).
const SLIDERS = ['tLo', 'tHi', 'leak', 'release', 'tDiv', 'divScale', 'gain']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()

$('run').addEventListener('click', () => run())
