// Background replacement pipelines, all with the same shape:
//
//   const p = await PIPELINES[name].create({ width, height })   // frame size of the clip
//   const out = await p.process(frame)   // VideoFrame in -> new VideoFrame out (green background)
//   p.infer(frame); await p.wait()       // model only (see bench.js "model" mode)
//
// Every pipeline does the same effect: replace the background with solid green,
// mix(green, frame, alpha), with the alpha bilinearly upsampled to the frame size,
// on the GPU, output as a new VideoFrame at the input's resolution.
//
// Structure, the MediaPipe Tasks pipelines and the LiteRT.js pipeline are adapted
// from github.com/sb2702/webrtc-hacks-webgpu (segmentation/web/src/pipelines.js,
// commit 2c6528c, MIT, (c) 2026 Sam Bhattacharyya).
//
// Rows (model x engine x pipeline), see benchmarks/METHODOLOGY.md:
//   1. mediapipe-*            MediaPipe selfie model, MediaPipe Tasks engine (GPU / CPU delegate)
//   2. litert-*               MediaPipe selfie model, LiteRT.js engine (WebGPU)
//   3. lp-mediapipe-*         MediaPipe selfie model, Longpipe engine (`longpipe/model` MediaPipeSelfie)
//   4. longpipe-*             Longpipe model, Longpipe engine, network on every frame (`longpipe/model`)
//   5. shipped-*              Longpipe model, Longpipe engine, the SDK's own per-frame renderer
//                             (key frames + learned optical-flow frames + stabiliser), as EffectsPipeline runs it

import {
  WebGPUBackend, WebGLBackend, TierModel, TIER_CONFIG, RenderOp, loadWeightsFromBinary,
  MediaPipeSelfie, MEDIAPIPE_SELFIE_CONFIG,
} from 'longpipe/model'
// Row 5 drives the SDK's internal Renderer (not exported by the package), compiled
// from this repo's src/ (vite.config.js); it needs the backends from the same source.
import { Renderer } from '~/pipeline/worker/renderer.ts'
import { PRESETS } from '~/pipeline/presets.ts'
import { WebGPUBackend as SrcWebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGLBackend as SrcWebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGPUFrames } from './frames/webgpu.js'
import { WebGLFrames, GLHelper } from './frames/webgl.js'

const GREEN = [0, 177 / 255, 64 / 255]

// ---- helpers ----------------------------------------------------------------

let devicePromise
function getDevice() {
  return (devicePromise ??= (async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) throw new Error('WebGPU is not available')
    const device = await adapter.requestDevice()
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
    return device
  })())
}

// Yield to the event loop without setTimeout's 1-4 ms clamping.
const channel = new MessageChannel()
const yieldTask = () => new Promise((resolve) => { channel.port1.onmessage = resolve; channel.port2.postMessage(0) })

// Wait until all GL commands issued so far have executed on the GPU. WebGL only
// updates sync status between tasks, so poll with a cheap yield.
async function glWait(gl) {
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
  gl.flush()
  for (;;) {
    const s = gl.clientWaitSync(fence, 0, 0)
    if (s === gl.ALREADY_SIGNALED || s === gl.CONDITION_SATISFIED) break
    if (s === gl.WAIT_FAILED) throw new Error('clientWaitSync failed')
    await yieldTask()
  }
  gl.deleteSync(fence)
}

const outputFrame = (canvas, frame) =>
  new VideoFrame(canvas, { timestamp: frame.timestamp, duration: frame.duration ?? undefined })

// ---- baseline ---------------------------------------------------------------

async function passthrough() {
  return { process: (frame) => frame, maskRes: null }
}

// Output-path ceilings: the same frame -> canvas -> VideoFrame path the effect
// pipelines use, but with no model (alpha = 1 everywhere, so the frame is copied
// through the composite shader). Shows how much of a pipeline's cost is just
// getting a canvas back into the encoder.
async function canvasCopyWebGPU() {
  const device = await getDevice()
  const frames = new WebGPUFrames(device)
  const ones = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(ones, 0, new Float32Array([1]))
  return {
    maskRes: null,
    process(frame) {
      const src = frames.import(frame)
      const encoder = device.createCommandEncoder()
      frames.composite(encoder, src, ones, { width: 1, height: 1, stride: 1 })
      device.queue.submit([encoder.finish()])
      return frames.output(frame)
    },
  }
}

async function canvasCopyWebGL() {
  const canvas = new OffscreenCanvas(1, 1)
  const gl = canvas.getContext('webgl2')
  gl.getExtension('EXT_color_buffer_float')
  const frames = new WebGLFrames(new GLHelper(gl), canvas)
  const ones = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, ones)
  // NEAREST: the default mipmapped min filter would leave the texture incomplete (reads 0).
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array([1]))
  return {
    maskRes: null,
    process(frame) {
      frames.upload(frame)
      frames.composite(ones, { width: 1, height: 1 })
      return frames.output(frame)
    },
  }
}

// ---- Longpipe, via the public model layer (`longpipe/model`) -----------------

// Precision. EffectsPipeline creates ONE backend for all presets, with dtype
// 'f16' by default (src/pipeline/index.ts:397), falling back to f32 only when
// WebGPU lacks shader-f16 (src/pipeline/worker/setup_backend.ts:61); the
// preset's own dtype in presets.ts (f32 for large/xl) is overridden by the
// resolved one (src/pipeline/worker/resolve_preset.ts:34). So as shipped every
// preset runs f16 where available. The `-f32` variants run large/xl at the
// precision presets.ts lists for them (what the 2026-11-15 run measured).
// Weights file: the one EffectsPipeline fetches for that precision
// (fetchModelWeights, src/pipeline/index.ts:207): .f16.bin for f16, .bin for f32.
const weightsFile = (tier, dtype) => `/assets/models/model_${tier}${dtype === 'f16' ? '.f16' : ''}.bin`

async function longpipe(kind, tier, { width, height }, requested = 'f16') {
  const cfg = TIER_CONFIG[tier]
  let dtype = requested
  if (kind === 'webgpu' && dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) dtype = 'f32'
  const file = weightsFile(tier, dtype)
  const weights = loadWeightsFromBinary(await (await fetch(file)).arrayBuffer())

  // The backend's canvas is the output: frame-sized, like every other pipeline.
  const canvas = new OffscreenCanvas(width, height)
  const backend = kind === 'webgpu'
    ? await WebGPUBackend.create({ canvas, dtype })
    : await WebGLBackend.create({ canvas, dtype })
  // Network input at the tier's canvas resolution (landscape; the clip is 16:9,
  // the Input op stretches, as MediaPipe does).
  const input = backend.ops.Input(cfg.canvasRes.h, cfg.canvasRes.w)
  const model = new TierModel(backend, input.output, weights.base, weights.wrapper, cfg.wrapper, cfg.base)
  const render = new RenderOp(backend)
  render.attachNetwork(model, input, { upscaler: 'bilinear', background: { mode: 'solid', color: GREEN } })

  return {
    dtype,
    maskRes: `${cfg.canvasRes.w}x${cfg.canvasRes.h}`,
    process(frame) {
      render.setSource(frame)
      render.run()          // network -> bilinear alpha upscale -> composite, every frame
      return outputFrame(canvas, frame)
    },
    infer(frame) {
      input.setSource(frame)
      input.run()           // resize frame -> network input
      model.run()           // network -> alpha at canvasRes
    },
    wait: () => (kind === 'webgpu' ? backend.sync() : glWait(backend.gl)),
    destroy: () => backend.destroy(),
  }
}

// ---- MediaPipe selfie model on Longpipe's engine (`longpipe/model`) -----------
// Same backend / Input op / RenderOp path as longpipe() above; only the network
// differs. This is exactly what EffectsPipeline({ model: 'mediapipe-selfie' })
// runs per frame (Renderer.setMediaPipeSelfie, src/pipeline/worker/renderer.ts:601:
// no frame skipping, no flow; RenderOp.runModel + runDisplay every frame).
// Weights: Longpipe's conversion of the same .tflite files (f16 packs from
// DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL), at the SDK's default backend precision.
// MediaPipe selfie on Longpipe's engine (added with 0.0.17).
async function longpipeMediaPipe(kind, variant, { width, height }, requested = 'f16') {
  const cfg = MEDIAPIPE_SELFIE_CONFIG[variant]
  let dtype = requested
  if (kind === 'webgpu' && dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) dtype = 'f32'
  const weights = loadWeightsFromBinary(await (await fetch(`/assets/models/${cfg.file}`)).arrayBuffer())
  const canvas = new OffscreenCanvas(width, height)
  const backend = kind === 'webgpu'
    ? await WebGPUBackend.create({ canvas, dtype })
    : await WebGLBackend.create({ canvas, dtype })
  const input = backend.ops.Input(cfg.inputRes.h, cfg.inputRes.w)
  const model = new MediaPipeSelfie(backend, input.output, weights)
  const render = new RenderOp(backend)
  render.attachNetwork(model, input, { upscaler: 'bilinear', background: { mode: 'solid', color: GREEN } })
  return {
    dtype,
    maskRes: `${cfg.inputRes.w}x${cfg.inputRes.h}`,
    process(frame) { render.setSource(frame); render.run(); return outputFrame(canvas, frame) },
    infer(frame) { input.setSource(frame); input.run(); model.run() },
    wait: () => (kind === 'webgpu' ? backend.sync() : glWait(backend.gl)),
    destroy: () => backend.destroy(),
  }
}

// ---- Longpipe as shipped: the SDK's own per-frame renderer ------------------
// The worker's Renderer (src/pipeline/worker/renderer.ts) with a fixed preset:
// what EffectsPipeline runs per frame in Chrome, where its output adapter is
// `renderer.process(frame); new VideoFrame(renderer.canvas)`
// (src/pipeline/worker/adapters/output_mstg.ts:15-16). Not included: the worker
// thread and MediaStreamTrack transports (every pipeline here runs on the page's
// main thread, fed by the same decoder), autotune and the adaptive controller
// (a fixed preset disables both: src/pipeline/index.ts:468), touch-up/reframe (off).
//
// Per frame, with the shipped weights (all carry a `flow` blob, so the flow path
// is wired: renderer.ts:653), Renderer.stepFlow (renderer.ts:382) runs:
//   skipFrames > 0 (xs: 2, small/medium: 1; presets.ts:30-34)
//     key frame (every skipFrames+1-th; shouldRunModel, renderer.ts:532):
//       full network (TierModel), flow net NOT run (cached flow reused)
//     off frame: learned OpticalFlowNet (last key frame -> current, on the base-res
//       frame pair + the key frame's cached encoder taps) + warp of the last key
//       frame's alpha
//   skipFrames = 0 (large, xl): flow net AND full network, every frame
//   then on every frame: warp the previous stabilised alpha, flow-gated
//   stabiliser blend, re-upscale (applyAlpha), composite.
// `counters` counts full-network and flow-network runs to verify this.
async function shipped(kind, tier, { width, height }) {
  // EffectsPipeline's default backend precision, patched into the preset as
  // resolve_preset.ts:34 does (see weightsFile above).
  let dtype = 'f16'
  if (kind === 'webgpu' && !(await SrcWebGPUBackend.hasF16Support())) dtype = 'f32'
  const preset = { ...PRESETS.find((p) => p.model === tier), dtype }
  const canvas = new OffscreenCanvas(width, height)
  const backend = kind === 'webgpu'
    ? await SrcWebGPUBackend.create({ canvas, dtype })
    : SrcWebGLBackend.create({ canvas, dtype })
  const renderer = new Renderer({
    backend, backendKind: kind, canvas,
    background: { kind: 'color', rgb: GREEN }, enabled: true,
    topology: { input: 'mstp', output: 'mstg' },   // only reported in stats
  })
  renderer.setPreset(preset, await (await fetch(weightsFile(tier, dtype))).arrayBuffer())

  // Instrumentation: count calls into the real objects the renderer built.
  const counts = { frames: 0, network: 0, flow: 0 }
  const wrap = (obj, key) => {
    if (!obj) return
    const run = obj.run.bind(obj)
    obj.run = () => { counts[key]++; run() }
  }
  wrap(renderer.renderOp.network, 'network')   // TierModel: the full matting network
  wrap(renderer.flow?.net, 'flow')             // OpticalFlowNet
  return {
    dtype,
    maskRes: `${TIER_CONFIG[tier].canvasRes.w}x${TIER_CONFIG[tier].canvasRes.h}`,
    temporal: true,
    preset: { model: preset.model, dtype: preset.dtype, skipFrames: preset.skipFrames, flowWired: !!renderer.flow },
    process(frame) {
      counts.frames++
      renderer.process(frame)
      return outputFrame(canvas, frame)
    },
    counters: {
      reset: () => { counts.frames = counts.network = counts.flow = 0 },
      read: () => ({ ...counts }),
    },
    destroy: () => renderer.destroy(),
  }
}

// ---- MediaPipe selfie model on LiteRT.js (WebGPU) -----------------------------
// LiteRT.js (Google's TFLite web runtime) runs the original .tflite on WebGPU,
// sharing our GPUDevice: the frame is resized into a dense [1, H, W, 3] f32
// buffer by a WebGPU compute pass (4-tap box filter, from the article harness),
// LiteRT runs on that buffer and hands back its output buffer, which our WebGPU
// shader composites (bilinear upsample). No readback. Default precision (fp32).
let liteRtPromise
async function getLiteRt() {
  return (liteRtPromise ??= (async () => {
    const litert = await import('@litertjs/core')
    await litert.loadLiteRt('/vendor/litert/', { threads: self.crossOriginIsolated })
    litert.setWebGpuDevice(await getDevice())
    return litert
  })())
}

async function liteRt(variant) {
  const [litert, device] = await Promise.all([getLiteRt(), getDevice()])
  const { url, res } = MP_MODELS[variant]
  const [W, H] = res.split('x').map(Number)
  const model = await litert.loadAndCompile(url, { accelerator: 'webgpu' })
  const frames = new WebGPUFrames(device)
  frames.setColor(GREEN)
  const inputBuffer = device.createBuffer({ size: W * H * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  const inputTensor = new litert.Tensor(inputBuffer, [1, H, W, 3], 'float32')
  const preprocess = (frame) => {
    const src = frames.import(frame)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    frames.preprocess(pass, src, inputBuffer, { width: W, height: H, stride: 3 })
    pass.end()
    device.queue.submit([encoder.finish()])
    return src
  }
  let last
  return {
    maskRes: res,
    async process(frame) {
      const src = preprocess(frame)
      const [out] = await model.run([inputTensor])
      const encoder = device.createCommandEncoder()
      frames.composite(encoder, src, out.toGpuBuffer(), { width: W, height: H, stride: 1 })
      device.queue.submit([encoder.finish()])
      out.delete()
      return frames.output(frame)
    },
    infer(frame) {
      preprocess(frame)
      last = model.run([inputTensor]).then(([out]) => out.delete())
    },
    wait: async () => { await last; await device.queue.onSubmittedWorkDone() },
    destroy: () => model.delete?.(),
  }
}

// ---- MediaPipe Tasks ImageSegmenter (its own pre/post-processing) -------------

const MP_MODELS = {
  square: { url: '/assets/models/selfie_segmenter.tflite', res: '256x256' },
  landscape: { url: '/assets/models/selfie_segmenter_landscape.tflite', res: '256x144' },
}

let filesetPromise
async function getVisionTasks() {
  return (filesetPromise ??= (async () => {
    const vision = await import('@mediapipe/tasks-vision')
    return { vision, fileset: await vision.FilesetResolver.forVisionTasks('/vendor/mediapipe') }
  })())
}

const segmenterOptions = (model, delegate, canvas) => ({
  baseOptions: { modelAssetPath: MP_MODELS[model].url, delegate },
  runningMode: 'VIDEO',
  outputConfidenceMasks: true,
  outputCategoryMask: false,
  ...(canvas && { canvas }),
})

// VIDEO mode wants strictly increasing timestamps; benchmarks replay the same
// clip, so count instead of using frame.timestamp.
const clock = () => { let t = 0; return () => ++t }

// GPU delegate = WebGL. MediaPipe renders into our canvas's WebGL2 context, so
// its mask texture is composited directly in that context, no readback.
async function mediaPipeGPU(model) {
  const { vision, fileset } = await getVisionTasks()
  const canvas = new OffscreenCanvas(1, 1)
  const segmenter = await vision.ImageSegmenter.createFromOptions(fileset, segmenterOptions(model, 'GPU', canvas))
  const gl = canvas.getContext('webgl2')
  const frames = new WebGLFrames(new GLHelper(gl), canvas)
  frames.setColor(GREEN)
  const now = clock()
  const p = {
    maskRes: MP_MODELS[model].res,
    process(frame) {
      const result = segmenter.segmentForVideo(frame, now())
      const mask = result.confidenceMasks.at(-1)
      p.deliveredMaskRes = `${mask.width}x${mask.height}`
      frames.upload(frame)
      frames.composite(mask.getAsWebGLTexture(), { width: mask.width, height: mask.height })
      gl.bindVertexArray(null)
      const out = frames.output(frame)
      result.close()
      return out
    },
    infer(frame) {
      const result = segmenter.segmentForVideo(frame, now())
      result.confidenceMasks.at(-1).getAsWebGLTexture()
      result.close()
    },
    wait: () => glWait(gl),
    destroy: () => segmenter.close(),
  }
  return p
}

// CPU delegate = WASM (XNNPACK). MediaPipe returns the mask as a Float32Array
// at the frame's resolution; it is uploaded and composited on WebGPU.
async function mediaPipeCPU(model) {
  const { vision, fileset } = await getVisionTasks()
  const segmenter = await vision.ImageSegmenter.createFromOptions(fileset, segmenterOptions(model, 'CPU'))
  const device = await getDevice()
  const frames = new WebGPUFrames(device)
  frames.setColor(GREEN)
  let maskBuffer
  const now = clock()
  const p = {
    maskRes: MP_MODELS[model].res,
    process(frame) {
      const result = segmenter.segmentForVideo(frame, now())
      const mask = result.confidenceMasks.at(-1)
      p.deliveredMaskRes = `${mask.width}x${mask.height}`
      const data = mask.getAsFloat32Array()
      if (maskBuffer?.size !== data.byteLength) {
        maskBuffer?.destroy()
        maskBuffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
      }
      device.queue.writeBuffer(maskBuffer, 0, data)
      const src = frames.import(frame)
      const encoder = device.createCommandEncoder()
      frames.composite(encoder, src, maskBuffer, { width: mask.width, height: mask.height, stride: 1 })
      device.queue.submit([encoder.finish()])
      result.close()
      return frames.output(frame)
    },
    infer(frame) {
      const result = segmenter.segmentForVideo(frame, now())   // synchronous: inference is done on return
      result.confidenceMasks.at(-1).getAsFloat32Array()
      result.close()
    },
    wait: async () => {},
    destroy: () => segmenter.close(),
  }
  return p
}

// ---- registry ---------------------------------------------------------------

export const TIERS = ['xs', 'small', 'medium', 'large', 'xl']

export const PIPELINES = {
  'passthrough': { label: 'Passthrough (decode → encode, no effect)', create: passthrough },
  'canvas-copy-webgpu': { label: 'Canvas copy, WebGPU (no model: output-path ceiling)', create: canvasCopyWebGPU },
  'canvas-copy-webgl': { label: 'Canvas copy, WebGL2 (no model: output-path ceiling)', create: canvasCopyWebGL },
  // 1. MediaPipe selfie model, MediaPipe Tasks engine, every frame
  'mediapipe-gpu': { label: 'MediaPipe square — Tasks, GPU delegate (WebGL)', create: () => mediaPipeGPU('square') },
  'mediapipe-gpu-landscape': { label: 'MediaPipe landscape — Tasks, GPU delegate (WebGL)', create: () => mediaPipeGPU('landscape') },
  'mediapipe-cpu': { label: 'MediaPipe square — Tasks, CPU delegate (WASM)', create: () => mediaPipeCPU('square') },
  'mediapipe-cpu-landscape': { label: 'MediaPipe landscape — Tasks, CPU delegate (WASM)', create: () => mediaPipeCPU('landscape') },
  // 2. MediaPipe selfie model, LiteRT.js engine, every frame
  'litert-webgpu': { label: 'MediaPipe square — LiteRT.js, WebGPU (fp32)', create: () => liteRt('square') },
  'litert-webgpu-landscape': { label: 'MediaPipe landscape — LiteRT.js, WebGPU (fp32)', create: () => liteRt('landscape') },
  // 3. MediaPipe selfie model, Longpipe engine, every frame
  ...Object.fromEntries(['webgpu', 'webgl'].flatMap((k) => ['square', 'landscape'].map((v) => [`lp-mediapipe-${k}-${v}`, {
    label: `MediaPipe ${v} — Longpipe engine, ${k === 'webgpu' ? 'WebGPU' : 'WebGL2'} (f16)`, create: (size) => longpipeMediaPipe(k, v, size),
  }]))),
  // 4. Longpipe model, Longpipe engine, network every frame
  ...Object.fromEntries(['webgpu', 'webgl'].flatMap((k) => TIERS.map((t) => [`longpipe-${k}-${t}`, {
    label: `Longpipe ${t} — network every frame, ${k === 'webgpu' ? 'WebGPU' : 'WebGL2'} (f16)`, create: (size) => longpipe(k, t, size, 'f16'),
  }]))),
  ...Object.fromEntries(['webgpu', 'webgl'].flatMap((k) => ['large', 'xl'].map((t) => [`longpipe-${k}-${t}-f32`, {
    label: `Longpipe ${t} — network every frame, ${k === 'webgpu' ? 'WebGPU' : 'WebGL2'} (f32)`, create: (size) => longpipe(k, t, size, 'f32'),
  }]))),
  // 5. Longpipe model, Longpipe engine, as shipped (SDK renderer: key + flow frames)
  ...Object.fromEntries(['webgpu', 'webgl'].flatMap((k) => TIERS.map((t) => [`shipped-${k}-${t}`, {
    label: `Longpipe ${t} — as shipped, ${k === 'webgpu' ? 'WebGPU' : 'WebGL2'} (f16)`, create: (size) => shipped(k, t, size),
  }]))),
}
