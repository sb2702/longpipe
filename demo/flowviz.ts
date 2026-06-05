import type { Backend, Dtype } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Optical-flow diagnostic: matting + flow run EVERY frame on the same checkpoints
// the production flow path uses (the SDK OpticalFlowNet is fidelity-exact to them),
// so this sidesteps the diverged training flow_model.py. Three live panels:
//   input | flow field (HSV: hue=direction, value=magnitude) | greenscreen
// The flow field is the raw base/4 OpticalFlowNet output, read back + drawn.

const DEC_W = 16
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s }

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

    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    // Flow stem input = CANVAS frame downsampled to base (matches training:
    // x_base = interpolate(canvas → base)). NOT a direct source→base resample —
    // that's a single ~5× bilinear pass and aliases, garbling the flow at edges.
    const curBaseDown = backend.ops.BilinearUpsample(netInput.output, { outH: b.h, outW: b.w })
    const frameAHeld = backend.tensor(b.h, b.w, 4, new Float32Array(b.h * b.w * 4))
    const flow = new OpticalFlowNet(backend, frameAHeld, curBaseDown.output, matting.encoderTaps, w.flow, DEC_W,
      tier === 'xs' ? { fuseStem: true, halfTap: matting.halfTap } : {})
    const greenComp = backend.presenters.CompositeSolid(netInput.output, matting.output, [0, 1, 0])

    const fw = flow.output.w, fh = flow.output.h
    flowC.width = fw; flowC.height = fh
    const flowCtx = flowC.getContext('2d')!
    const flowImg = flowCtx.createImageData(fw, fh)

    status(`running — input | flow (base/4 ${fw}×${fh}) | greenscreen`)
    const tick = async () => {
      if (!running) return
      const bm = await createImageBitmap(video)
      netInput.setSource(bm)
      netInput.run()
      curBaseDown.run()                            // canvas → base (matches training stem input)
      flow.run()                                   // flow before matting → taps are prev's
      matting.run()
      backend.copyTensor(curBaseDown.output, frameAHeld)
      greenComp.run()

      const fd = await backend.readback(flow.output)
      const gain = parseFloat($<HTMLInputElement>('gain').value)
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

$('run').addEventListener('click', () => run())
