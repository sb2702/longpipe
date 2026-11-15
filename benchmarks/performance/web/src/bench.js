// End-to-end benchmark: decode -> background replacement -> encode -> mux, as
// fast as the pipeline allows (no real-time pacing), for each pipeline; plus a
// model-only latency mode.
//
// Adapted from github.com/sb2702/webrtc-hacks-webgpu (segmentation/web/src/bench.js,
// MIT). What is and isn't timed (see benchmarks/METHODOLOGY.md):
//   - The file is demuxed once up front; each run replays the cached
//     EncodedVideoChunks, so demuxing is excluded.
//   - Pipeline creation (weights fetch from the local dev server, shader /
//     graph compilation) is excluded and reported separately as initMs.
//   - Each pipeline gets one untimed warm-up pass over the clip.
//   - A timed transcode run is wall-clock time from the first chunk entering
//     the decoder to the muxer finishing.
//   - "model" mode times only the model (see runModel), per frame, GPU-synchronised.

import {
  SimpleDemuxer, SimpleMuxer, VideoDecodeStream, VideoProcessStream, VideoEncodeStream, getBitrate, getCodecString,
} from 'webcodecs-utils'
import { PIPELINES } from './pipelines.js'

// web-demuxer fetches this from a worker, so it needs an absolute URL.
const DEMUXER_WASM = new URL('/vendor/web-demuxer.wasm', location.href).href

const demuxed = new WeakMap()
export function demux(file) {
  if (!demuxed.has(file)) {
    demuxed.set(file, (async () => {
      const demuxer = new SimpleDemuxer(file, { wasmFilePath: DEMUXER_WASM })
      await demuxer.load()
      const config = await demuxer.getVideoDecoderConfig()
      const chunks = []
      await demuxer.videoStream().pipeTo(new WritableStream({ write: (chunk) => { chunks.push(chunk) } }))
      return { config, chunks }
    })())
  }
  return demuxed.get(file)
}

const chunkStream = (chunks) => new ReadableStream({
  start(controller) {
    for (const chunk of chunks) controller.enqueue(chunk)
    controller.close()
  },
})

const sorted = (xs) => [...xs].sort((a, b) => a - b)
const median = (xs) => {
  const s = sorted(xs), m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
const percentile = (xs, p) => sorted(xs)[Math.min(xs.length - 1, Math.floor(xs.length * p))]

// The H.264 settings every pipeline's output is encoded with.
export function encoderConfig(width, height) {
  const bitrate = getBitrate(width, height, 30, 'good')
  return { codec: getCodecString('avc', width, height, bitrate), width, height, bitrate }
}

// The output stage between process() and the encoder, the same for every
// pipeline (passthrough included). 'direct' hands the pipeline's VideoFrame to
// the encoder as is. 'readback' copies each frame's pixels to CPU memory with
// VideoFrame.copyTo() (async, up to `depth` frames in flight, output kept in
// order) and gives the encoder a new RGBA VideoFrame over that buffer.
//
// Why: in Chrome 157 on the M4 tested, the encoder takes ~10 ms per frame to
// consume a canvas-backed VideoFrame (any canvas type, any encoder setting),
// which capped every effect pipeline at ~80-150 fps, while copyTo() on the same
// frames runs at ~800 fps (360p) and the encoder takes CPU RGBA frames at
// >1,000 fps. See METHODOLOGY.md, "Output path".
export function readbackStream(depth = 4) {
  const pending = []
  return new TransformStream({
    async transform(frame, controller) {
      const w = frame.displayWidth, h = frame.displayHeight
      const buf = new Uint8Array(w * h * 4)
      pending.push(frame.copyTo(buf, { format: 'RGBA' }).then(() => {
        const out = new VideoFrame(buf, { format: 'RGBA', codedWidth: w, codedHeight: h, timestamp: frame.timestamp, duration: frame.duration ?? undefined })
        frame.close()
        return out
      }))
      if (pending.length >= depth) controller.enqueue(await pending.shift())
    },
    async flush(controller) {
      while (pending.length) controller.enqueue(await pending.shift())
    },
  })
}
export const OUTPUT_PATHS = ['readback', 'direct']

// Forces a frame's GPU work to finish without reading back the whole frame:
// draw it into a 1x1 (GPU-backed) canvas and read that single pixel.
const probe = new OffscreenCanvas(1, 1).getContext('2d')
function waitForPixels(frame) {
  probe.drawImage(frame, 0, 0, 1, 1)
  probe.getImageData(0, 0, 1, 1)
}

// One pass over the clip.
// mode 'transcode': decode -> process -> output stage -> encode -> mux (throughput).
// mode 'process':   decode -> process -> wait for the output's pixels, one frame
//                   at a time, no encoder (per-frame latency of the whole effect).
export async function runOnce(pipeline, file, mode = 'transcode', output = 'readback') {
  const { config: decoderConfig, chunks } = await demux(file)
  const { codedWidth: width, codedHeight: height } = decoderConfig
  const latencies = []
  const start = performance.now()
  const processed = chunkStream(chunks)
    .pipeThrough(new VideoDecodeStream(decoderConfig))
    .pipeThrough(new VideoProcessStream(async (frame) => {
      const t = performance.now()
      const out = await pipeline.process(frame)
      if (mode === 'process') waitForPixels(out)
      latencies.push(performance.now() - t)
      return out
    }))
  if (mode === 'transcode') {
    const muxer = new SimpleMuxer({ video: 'avc' })
    await processed
      .pipeThrough(output === 'readback' ? readbackStream() : new TransformStream())
      .pipeThrough(new VideoEncodeStream(encoderConfig(width, height)))
      .pipeTo(muxer.videoSink())
    await muxer.finalize()
  } else {
    await processed.pipeTo(new WritableStream({ write: (frame) => frame.close() }))
  }
  const ms = performance.now() - start
  return {
    frames: latencies.length,
    ms,
    fps: (latencies.length * 1000) / ms,
    // Time inside process() per frame. In transcode mode, for GPU pipelines this
    // is CPU-side submission cost, not GPU time; in process mode it includes
    // waiting for the output's pixels.
    meanProcessMs: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    medianProcessMs: median(latencies),
  }
}

// Decode the first n frames of the clip and keep them (for model mode / checks).
async function decodeFrames(file, n) {
  const { config, chunks } = await demux(file)
  const frames = []
  await chunkStream(chunks).pipeThrough(new VideoDecodeStream(config)).pipeTo(new WritableStream({
    write(frame) { if (frames.length < n) frames.push(frame); else frame.close() },
  }))
  return frames
}

// Model-only latency: for each frame, submit the model work (pipeline.infer)
// and wait until the GPU has finished it (pipeline.wait), one frame at a time.
// Longpipe: resize the frame to the network input + network, alpha left at the
// tier's canvas resolution (no upscale, no composite).
// MediaPipe: segmentForVideo(), which includes MediaPipe's own resize, the
// network and its mask upsampling to frame size (not separable via its API).
export async function runModel(pipeline, frames, iterations) {
  const ms = []
  for (let i = 0; i < iterations; i++) {
    const t = performance.now()
    pipeline.infer(frames[i % frames.length])
    await pipeline.wait()
    ms.push(performance.now() - t)
  }
  return { iterations, medianMs: median(ms), p90Ms: percentile(ms, 0.9), meanMs: ms.reduce((a, b) => a + b, 0) / ms.length }
}

// One measurement of one pipeline, for tools/bench.mjs, which interleaves
// pipelines across rounds and repeats any measurement that overlapped background
// load: create -> warm-up (one untimed pass over the clip, or 30 model
// iterations) -> one timed pass (or `modelIterations`) -> destroy.
// t0/t1 (epoch ms) bracket the timed part.
export async function runOne(name, file, { mode = 'transcode', output = 'readback', modelIterations = 200 } = {}) {
  const { config } = await demux(file)
  const size = { width: config.codedWidth, height: config.codedHeight }
  const r = { name, label: PIPELINES[name].label }
  let pipeline, frames
  try {
    const t = performance.now()
    pipeline = await PIPELINES[name].create(size)
    r.initMs = performance.now() - t
    r.maskRes = pipeline.maskRes
    if (pipeline.dtype) r.dtype = pipeline.dtype
    if (pipeline.preset) r.preset = pipeline.preset
    if (mode === 'model') {
      if (!pipeline.infer) { r.skipped = 'no model-only mode'; return r }
      frames = await decodeFrames(file, 48)
      await runModel(pipeline, frames, 30)
      r.t0 = Date.now()
      Object.assign(r, await runModel(pipeline, frames, modelIterations))
    } else {
      r.warmupFps = (await runOnce(pipeline, file, mode, output)).fps
      pipeline.counters?.reset()
      r.t0 = Date.now()
      Object.assign(r, await runOnce(pipeline, file, mode, output))
      if (pipeline.counters) r.counts = pipeline.counters.read()
    }
    r.t1 = Date.now()
    if (pipeline.deliveredMaskRes) r.deliveredMaskRes = pipeline.deliveredMaskRes
  } catch (e) {
    console.error(e)
    r.error = String(e?.message ?? e)
  } finally {
    frames?.forEach((f) => f.close())
    try { pipeline?.destroy?.() } catch (e) { console.error(e) }
  }
  return r
}

// Benchmark several pipelines on one file, sequentially: each pipeline in turn
// is created, warmed up (one untimed pass), timed `runs` times, destroyed.
export async function benchmarkAll(names, file, { mode = 'transcode', output = 'readback', runs = 5, modelIterations = 200, onProgress, onResult } = {}) {
  const { config } = await demux(file)
  const size = { width: config.codedWidth, height: config.codedHeight }
  const frames = mode === 'model' ? await decodeFrames(file, 48) : null
  const results = []
  for (const name of names) {
    const r = { name, label: PIPELINES[name].label, runs: [], startedAt: new Date().toISOString() }
    results.push(r)
    try {
      const t = performance.now()
      const pipeline = await PIPELINES[name].create(size)
      r.initMs = performance.now() - t
      r.maskRes = pipeline.maskRes
      if (pipeline.dtype) r.dtype = pipeline.dtype
      if (mode !== 'model') {
        onProgress?.(`${name}: warming up…`)
        r.warmupFps = (await runOnce(pipeline, file, mode, output)).fps
        for (let i = 0; i < runs; i++) {
          onProgress?.(`${name}: run ${i + 1}/${runs}`)
          r.runs.push(await runOnce(pipeline, file, mode, output))
        }
        const fps = r.runs.map((x) => x.fps)
        Object.assign(r, { medianFps: median(fps), minFps: Math.min(...fps), maxFps: Math.max(...fps) })
      } else if (pipeline.infer) {
        onProgress?.(`${name}: warming up…`)
        await runModel(pipeline, frames, 30)
        for (let i = 0; i < runs; i++) {
          onProgress?.(`${name}: run ${i + 1}/${runs}`)
          r.runs.push(await runModel(pipeline, frames, modelIterations))
        }
        const m = r.runs.map((x) => x.medianMs)
        Object.assign(r, { medianMs: median(m), minMs: Math.min(...m), maxMs: Math.max(...m) })
      }
      if (pipeline.deliveredMaskRes) r.deliveredMaskRes = pipeline.deliveredMaskRes
      r.endedAt = new Date().toISOString()
      pipeline.destroy?.()
    } catch (e) {
      console.error(e)
      r.error = String(e?.message ?? e)
    }
    onResult?.(r)
  }
  frames?.forEach((f) => f.close())
  return {
    config: {
      mode, output, runs, order: 'sequential', file: file.name, frames: results.find((r) => r.runs.length)?.runs[0].frames,
      ...(mode === 'model' && { modelIterations }),
      width: config.codedWidth, height: config.codedHeight, decoder: config.codec,
      ...(mode === 'transcode' && { encoder: encoderConfig(config.codedWidth, config.codedHeight) }),
    },
    environment: await environment(),
    results,
  }
}

// Everything about the machine a reader would need to compare results.
export async function environment() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null)
  const ua = await navigator.userAgentData?.getHighEntropyValues?.(['platform', 'platformVersion', 'architecture', 'model', 'fullVersionList']).catch(() => null)
  const gl = new OffscreenCanvas(1, 1).getContext('webgl2')
  const dbg = gl?.getExtension('WEBGL_debug_renderer_info')
  return {
    date: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: ua && { platform: ua.platform, platformVersion: ua.platformVersion, architecture: ua.architecture, model: ua.model, browsers: ua.fullVersionList },
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGB: navigator.deviceMemory,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: adapter && {
      vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device,
      description: adapter.info.description, features: [...adapter.features].sort(),
    },
    webglRenderer: gl && (dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)),
  }
}

// Run each pipeline on one frame and summarise its output, to catch broken
// output (flipped, inverted, empty masks) before trusting the timings.
export async function checkOutputs(names, file, frameIndex = 96) {
  const { config } = await demux(file)
  const size = { width: config.codedWidth, height: config.codedHeight }
  const frames = await decodeFrames(file, frameIndex + 1)
  const target = frames.pop()
  const history = frames.splice(-12)
  frames.forEach((f) => f.close())
  const out = {}
  for (const name of names) {
    try {
      const pipeline = await PIPELINES[name].create(size)
      // Temporal pipelines (as shipped) see the 12 preceding frames first, so the
      // checked frame goes through the flow/stabiliser path with real history.
      if (pipeline.temporal) for (const f of history) { const o = await pipeline.process(f); if (o !== f) o.close() }
      const frame = await pipeline.process(target)
      const bitmap = await createImageBitmap(frame)
      if (frame !== target) frame.close()
      pipeline.destroy?.()
      const canvas = new OffscreenCanvas(bitmap.width / 4, bitmap.height / 4)
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
      out[name] = { bitmap, pixels: ctx.getImageData(0, 0, canvas.width, canvas.height).data }
    } catch (e) {
      console.error(e)
      out[name] = { error: String(e?.message ?? e) }
    }
  }
  target.close()
  history.forEach((f) => f.close())
  // Background share = pixels that are (nearly) the green fill color.
  const isGreen = (d, j) => Math.abs(d[j]) < 12 && Math.abs(d[j + 1] - 177) < 12 && Math.abs(d[j + 2] - 64) < 12
  const ref = out['mediapipe-gpu']?.pixels
  for (const r of Object.values(out)) {
    if (r.error) continue
    let green = 0, agree = 0
    const n = r.pixels.length / 4
    for (let j = 0; j < r.pixels.length; j += 4) {
      const g = isGreen(r.pixels, j)
      green += g
      if (ref) agree += g === isGreen(ref, j)
    }
    r.background = green / n
    r.agreement = ref ? agree / n : null
  }
  return out
}
