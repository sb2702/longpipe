// Output-path investigation: why does a canvas-rendered frame cost so much more
// to encode than a decoded frame? Every variant here has NO model: it only moves
// the decoded frame to the H.264 encoder by a different route, so the FPS is the
// ceiling that route puts on any effect pipeline.
//
// See benchmarks/METHODOLOGY.md ("Output path") for what was found.

import { SimpleMuxer, VideoDecodeStream, VideoEncodeStream, VideoProcessStream } from 'webcodecs-utils'
import { demux, encoderConfig, readbackStream } from './bench.js'
import { WebGPUFrames } from './frames/webgpu.js'
import { WebGLFrames, GLHelper } from './frames/webgl.js'

let devicePromise
const getDevice = () => (devicePromise ??= (async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  return adapter.requestDevice()
})())

const stamp = (frame) => ({ timestamp: frame.timestamp, duration: frame.duration ?? undefined })

// Each route: create() -> { process(frame) -> VideoFrame }
export const ROUTES = {
  // Decoded frame straight to the encoder.
  passthrough: async () => ({ process: (f) => f }),
  // A new VideoFrame wrapping the decoded frame's buffer (no pixels move).
  'clone': async () => ({ process: (f) => new VideoFrame(f, stamp(f)) }),
  // WebGPU full-screen pass into an OffscreenCanvas, then new VideoFrame(canvas):
  // what every effect pipeline in pipelines.js does.
  'webgpu-offscreen': async () => {
    const device = await getDevice()
    const frames = new WebGPUFrames(device)
    const ones = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(ones, 0, new Float32Array([1]))
    return {
      process(f) {
        const src = frames.import(f)
        const enc = device.createCommandEncoder()
        frames.composite(enc, src, ones, { width: 1, height: 1, stride: 1 })
        device.queue.submit([enc.finish()])
        return frames.output(f)
      },
    }
  },
  // Same, but the canvas is an on-page <canvas> element.
  'webgpu-dom-canvas': async () => {
    const device = await getDevice()
    const frames = new WebGPUFrames(device)
    const el = document.createElement('canvas')
    el.style.width = '64px'
    document.body.append(el)
    frames.canvas = el
    frames.context = el.getContext('webgpu')
    frames.context.configure({ device, format: frames.format, alphaMode: 'opaque' })
    const ones = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(ones, 0, new Float32Array([1]))
    return {
      process(f) {
        const src = frames.import(f)
        const enc = device.createCommandEncoder()
        frames.composite(enc, src, ones, { width: 1, height: 1, stride: 1 })
        device.queue.submit([enc.finish()])
        return new VideoFrame(el, stamp(f))
      },
      destroy: () => el.remove(),
    }
  },
  // WebGPU, then transferToImageBitmap -> new VideoFrame(bitmap).
  'webgpu-bitmap': async () => {
    const device = await getDevice()
    const frames = new WebGPUFrames(device)
    const ones = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(ones, 0, new Float32Array([1]))
    return {
      process(f) {
        const src = frames.import(f)
        const enc = device.createCommandEncoder()
        frames.composite(enc, src, ones, { width: 1, height: 1, stride: 1 })
        device.queue.submit([enc.finish()])
        const bmp = frames.canvas.transferToImageBitmap()
        const out = new VideoFrame(bmp, stamp(f))
        bmp.close()
        return out
      },
    }
  },
  // WebGL2 draw into an OffscreenCanvas, then new VideoFrame(canvas).
  'webgl-offscreen': async () => {
    const canvas = new OffscreenCanvas(1, 1)
    const gl = canvas.getContext('webgl2')
    gl.getExtension('EXT_color_buffer_float')
    const frames = new WebGLFrames(new GLHelper(gl), canvas)
    const ones = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, ones)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, new Float32Array([1]))
    return { process(f) { frames.upload(f); frames.composite(ones, { width: 1, height: 1 }); return frames.output(f) } }
  },
  // 2D canvas drawImage, then new VideoFrame(canvas).
  '2d-offscreen': async () => {
    const canvas = new OffscreenCanvas(1, 1)
    const ctx = canvas.getContext('2d')
    return {
      process(f) {
        if (canvas.width !== f.displayWidth) { canvas.width = f.displayWidth; canvas.height = f.displayHeight }
        ctx.drawImage(f, 0, 0)
        return new VideoFrame(canvas, stamp(f))
      },
    }
  },
  // CPU memory: the same RGBA buffer each frame (encoder must upload + convert).
  'cpu-rgba': async () => {
    let buf
    return {
      process(f) {
        const w = f.displayWidth, h = f.displayHeight
        buf ??= new Uint8Array(w * h * 4).fill(128)
        return new VideoFrame(buf, { format: 'RGBA', codedWidth: w, codedHeight: h, ...stamp(f) })
      },
    }
  },
  // CPU memory, NV12 (the encoder's native format: upload only, no conversion).
  'cpu-nv12': async () => {
    let buf
    return {
      process(f) {
        const w = f.displayWidth, h = f.displayHeight
        buf ??= new Uint8Array(w * h * 3 / 2).fill(128)
        return new VideoFrame(buf, { format: 'NV12', codedWidth: w, codedHeight: h, ...stamp(f) })
      },
    }
  },
}

// Encoder options to try on top of encoderConfig() (which leaves both unset).
export const ENCODER_VARIANTS = {
  default: {},
  'prefer-hardware': { hardwareAcceleration: 'prefer-hardware' },
  'prefer-software': { hardwareAcceleration: 'prefer-software' },
  realtime: { latencyMode: 'realtime' },
}

// decode -> route -> encode -> mux, the same stream chain as bench.js runOnce
// (webcodecs-utils streams), plus per-frame time spent inside process().
export async function transcode(route, file, encoderOpts = {}, readback = 0) {
  const { config, chunks } = await demux(file)
  const { codedWidth: width, codedHeight: height } = config
  const inProcess = []
  const t0 = performance.now()
  const muxer = new SimpleMuxer({ video: 'avc' })
  await new ReadableStream({ start(c) { chunks.forEach((x) => c.enqueue(x)); c.close() } })
    .pipeThrough(new VideoDecodeStream(config))
    .pipeThrough(new VideoProcessStream(async (f) => {
      const t = performance.now()
      const out = await route.process(f)
      inProcess.push(performance.now() - t)
      return out
    }))
    .pipeThrough(readback ? readbackStream(readback) : new TransformStream())
    .pipeThrough(new VideoEncodeStream({ ...encoderConfig(width, height), ...encoderOpts }))
    .pipeTo(muxer.videoSink())
  await muxer.finalize()
  const ms = performance.now() - t0
  inProcess.sort((a, b) => a - b)
  return { fps: (inProcess.length * 1000) / ms, frames: inProcess.length, medianProcessMs: inProcess[inProcess.length >> 1] }
}

// Encoder alone, no decoder and no stream helpers: `n` frames produced by
// `make(i)` are fed to a VideoEncoder with event-driven backpressure (wait on
// 'dequeue' when 20 frames are queued; no setTimeout polling). FPS = n / time
// to the last output chunk.
export async function encodeOnly(make, width, height, n = 192, encoderOpts = {}) {
  let outputs = 0
  const enc = new VideoEncoder({ output: () => { outputs++ }, error: (e) => console.error(e) })
  enc.configure({ ...encoderConfig(width, height), ...encoderOpts })
  const t0 = performance.now()
  for (let i = 0; i < n; i++) {
    while (enc.encodeQueueSize >= 20) await new Promise((r) => enc.addEventListener('dequeue', r, { once: true }))
    const f = await make(i)
    enc.encode(f, { keyFrame: i % 60 === 0 })
    f.close()
  }
  await enc.flush()
  const ms = performance.now() - t0
  enc.close()
  return { fps: (n * 1000) / ms, outputs }
}

// Readback throughput: WebGPU-rendered canvas -> new VideoFrame(canvas) ->
// frame.copyTo(RGBA buffer), with up to `depth` copies in flight. Tests whether
// an explicit async readback avoids the per-frame stall the encoder hits on
// canvas-backed frames. Returns frames/s of completed copies.
export async function readbackThroughput(width, height, n = 192, depth = 1) {
  const device = await getDevice()
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('webgpu')
  ctx.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' })
  const pending = []
  const t0 = performance.now()
  for (let i = 0; i < n; i++) {
    const enc = device.createCommandEncoder()
    enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [i / n, 0.5, 0.2, 1] }] }).end()
    device.queue.submit([enc.finish()])
    const frame = new VideoFrame(canvas, { timestamp: i })
    const buf = new Uint8Array(width * height * 4)
    pending.push(frame.copyTo(buf, { format: 'RGBA' }).finally(() => frame.close()))
    if (pending.length >= depth) await pending.shift()
  }
  await Promise.all(pending)
  return { fps: (n * 1000) / (performance.now() - t0), depth }
}
