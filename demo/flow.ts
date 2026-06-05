import type { Backend, Dtype, Tensor } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Standalone optical-flow temporal demo. Runs matting every N frames; on the
// in-between (skipped) frames the alpha is approximated by warping the last full
// inference forward along the flow predicted by OpticalFlowNet (riding the cached
// matting encoder taps). Validates the runtime chain end-to-end before the
// production renderer wiring. Source: /loop_video.mp4 (reproducible motion).
//
// Warp happens at the network-output (canvas) res, before the display upscale —
// flowScale = -(canvasW/baseW) folds the backward-gather negation + the base→
// canvas magnitude rescale (the flow predicts base-res-unit displacements).

const DEC_W = 16   // shipping flow width

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s; console.log('[flow]', s) }
const flowWidthName = (t: string) => `/model_${t}_flow`

async function createBackend(name: string, dtype: Dtype, canvas: HTMLCanvasElement): Promise<Backend> {
  if (name === 'webgpu') {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    if (dtype === 'f16' && !(await WebGPUBackend.hasF16Support()))
      throw new Error('adapter lacks shader-f16 — pick f32 or WebGL')
    return WebGPUBackend.create({ canvas, dtype })
  }
  return WebGLBackend.create({ canvas, dtype })
}

async function fetchBin(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url)
  if (!r.ok) return null
  if ((r.headers.get('content-type') ?? '').startsWith('text/html')) return null
  return r.arrayBuffer()
}

async function fetchWeights(tier: string, dtype: Dtype): Promise<ArrayBuffer> {
  const base = flowWidthName(tier)
  if (dtype === 'f16') {
    const f16 = await fetchBin(`${base}.f16.bin`)
    if (f16) return f16
  }
  const f32 = await fetchBin(`${base}.bin`)
  if (!f32) throw new Error(`failed to fetch ${base}.bin (run export_sdk_weights with --flow-checkpoint)`)
  return f32
}

async function loadVideo(url: string): Promise<HTMLVideoElement> {
  const v = document.createElement('video')
  v.src = url; v.loop = true; v.muted = true; v.playsInline = true
  await v.play()
  return v
}

let running = false

async function run() {
  if (running) return
  running = true
  try {
    const tier    = $<HTMLSelectElement>('tier').value
    const backend_ = $<HTMLSelectElement>('backend').value
    const dtype    = $<HTMLSelectElement>('dtype').value as Dtype
    const inferEvery = Math.max(1, parseInt($<HTMLInputElement>('inferEvery').value) || 3)

    const cfg = TIER_CONFIG[tier]
    const canvasW = cfg.canvasRes.w, canvasH = cfg.canvasRes.h
    const baseW = cfg.baseRes.w, baseH = cfg.baseRes.h
    const flowScale = -(canvasW / baseW)   // negate (backward gather) + base→canvas magnitude

    const out = $<HTMLCanvasElement>('outputCanvas')
    out.width = canvasW; out.height = canvasH

    status(`loading ${tier} weights (${backend_}/${dtype})…`)
    const [weightsBuf, video] = await Promise.all([fetchWeights(tier, dtype), loadVideo('/loop_video.mp4')])
    const backend = await createBackend(backend_, dtype, out)
    const w = loadWeightsFromBinary(weightsBuf) as any
    if (!w.flow) throw new Error('weights have no `flow` blob — re-export with --flow-checkpoint')

    // ── Build the graph once; per-frame run() reads whatever's in the buffers ──
    const netInput  = backend.ops.Input(canvasH, canvasW)               // x_hr (matting)
    const curBase   = backend.ops.Input(baseH, baseW)                    // current frame at base res (flow frame-b)
    const dispInput = backend.ops.Input(canvasH, canvasW)               // RGB for the compositor

    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)

    // Held state: the last full frame's base-RGB (flow frame-a) + its canvas-res alpha (warp source).
    const frameAHeld = backend.tensor(baseH, baseW, 4, new Float32Array(baseH * baseW * 4))
    const alphaHeld  = backend.tensor(canvasH, canvasW, 4, new Float32Array(canvasH * canvasW * 4))

    // XS rides a tap-half flow head (fuses the /2 tap at the stem → predicts base/2).
    const fuseStem = tier === 'xs'
    const flow = new OpticalFlowNet(backend, frameAHeld, curBase.output, matting.encoderTaps, w.flow, DEC_W,
      fuseStem ? { fuseStem: true, halfTap: matting.halfTap } : {})
    const flowUp = backend.ops.BilinearUpsample(flow.output, { outH: canvasH, outW: canvasW })
    const warp   = backend.ops.Warp(alphaHeld, flowUp.output, { flowScale })

    // The alpha the compositor shows: matting on full frames, warp on skips.
    const dispAlpha  = backend.tensor(canvasH, canvasW, 4, new Float32Array(canvasH * canvasW * 4))
    const compositor = backend.presenters.CompositeSolid(dispInput.output, dispAlpha, [0.04, 0.5, 0.27])

    status('running — green = flow-warped skip frames')
    let frameIdx = 0
    let fpsT = performance.now(), fpsN = 0

    const tick = async () => {
      if (!running) return
      const frame = await createImageBitmap(video)
      netInput.setSource(frame); curBase.setSource(frame); dispInput.setSource(frame)
      curBase.run(); dispInput.run()

      if (frameIdx % inferEvery === 0) {
        netInput.run()
        matting.run()
        backend.copyTensor(matting.output, alphaHeld)     // warp source = last full-frame alpha
        backend.copyTensor(curBase.output, frameAHeld)    // flow frame-a = last full frame
        backend.copyTensor(matting.output, dispAlpha)
      } else {
        flow.run(); flowUp.run(); warp.run()
        backend.copyTensor(warp.output, dispAlpha)
      }
      compositor.run()
      frame.close()

      frameIdx++
      if (++fpsN >= 30) {
        const now = performance.now()
        status(`${(fpsN * 1000 / (now - fpsT)).toFixed(0)} fps · infer 1/${inferEvery} · ${tier} ${backend_}/${dtype}`)
        fpsT = now; fpsN = 0
      }
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    running = false
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

$('run').addEventListener('click', () => { running = false; requestAnimationFrame(() => run()) })
