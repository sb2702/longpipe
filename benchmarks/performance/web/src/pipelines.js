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
// Structure and the MediaPipe pipelines are adapted from
// github.com/sb2702/webrtc-hacks-webgpu (segmentation/web/src/pipelines.js, MIT).

import {
  WebGPUBackend, WebGLBackend, TierModel, TIER_CONFIG, RenderOp, loadWeightsFromBinary,
} from 'longpipe/model'
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

// Same precision per tier as the SDK's production presets (src/pipeline/presets.ts).
// Override for experiments: ?dtype=f16 or ?dtype=f32 (npm run bench -- --dtype f16).
const PRESET_DTYPE = { xs: 'f16', small: 'f16', medium: 'f16', large: 'f32', xl: 'f32' }
const DTYPE_OVERRIDE = new URLSearchParams(location.search).get('dtype')
export const LONGPIPE_DTYPE = Object.fromEntries(Object.entries(PRESET_DTYPE).map(([t, d]) => [t, DTYPE_OVERRIDE ?? d]))

async function longpipe(kind, tier, { width, height }) {
  const cfg = TIER_CONFIG[tier]
  let dtype = LONGPIPE_DTYPE[tier]
  if (kind === 'webgpu' && dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) dtype = 'f32'
  // The weights file the SDK ships for the preset's precision; backends convert
  // at upload if the file's precision differs from the backend's.
  const file = `/assets/models/model_${tier}${PRESET_DTYPE[tier] === 'f16' ? '.f16' : ''}.bin`
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
  ...Object.fromEntries(TIERS.map((t) => [`longpipe-webgpu-${t}`, {
    label: `Longpipe ${t} — WebGPU (${LONGPIPE_DTYPE[t]})`, create: (size) => longpipe('webgpu', t, size),
  }])),
  ...Object.fromEntries(TIERS.map((t) => [`longpipe-webgl-${t}`, {
    label: `Longpipe ${t} — WebGL2 (${LONGPIPE_DTYPE[t]})`, create: (size) => longpipe('webgl', t, size),
  }])),
  'mediapipe-gpu': { label: 'MediaPipe selfie (square) — GPU (WebGL)', create: () => mediaPipeGPU('square') },
  'mediapipe-gpu-landscape': { label: 'MediaPipe selfie landscape — GPU (WebGL)', create: () => mediaPipeGPU('landscape') },
  'mediapipe-cpu': { label: 'MediaPipe selfie (square) — CPU (WASM)', create: () => mediaPipeCPU('square') },
  'mediapipe-cpu-landscape': { label: 'MediaPipe selfie landscape — CPU (WASM)', create: () => mediaPipeCPU('landscape') },
}
