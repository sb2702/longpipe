// WebCodecs perf benchmark — Longpipe vs MediaPipe Selfie Segmenter.
//
// Streams a test video through demux → decode → process(segment + composite)
// → encode. The encoder back-pressures the pipeline, so total wall clock
// divided by frames processed = real FPS, not bounded by display refresh.
// We do not finalize the muxer — the encoded frames are produced and dropped.
//
// Both backends run on WebGPU and produce VideoFrames at the source video's
// resolution with an image virtual background composited in. The Longpipe
// path uses the SDK's RenderOp end-to-end. The MediaPipe path runs the
// official `@mediapipe/tasks-vision` ImageSegmenter (with GPU delegate when
// available), uploads the resulting CPU mask to a WebGPU storage buffer,
// and runs a small fullscreen-quad shader for the upscale + composite.

import {
  SimpleDemuxer,
  VideoDecodeStream,
  VideoProcessStream,
  VideoEncodeStream,
  SimpleMuxer,
  // @ts-expect-error — esm.sh URL has no .d.ts
} from 'https://esm.sh/webcodecs-utils'

// MediaPipe types are pulled in dynamically below to keep TS happy.

import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { EfficientNetLiteMattingXL }      from '~/model/networks/efficientnetlite_matting_xl'
import { EfficientNetLiteMattingLarge }   from '~/model/networks/efficientnetlite_matting_large'
import { EfficientNetLiteMattingCompact } from '~/model/networks/efficientnetlite_matting_compact'
import { EfficientNetLiteMattingSmall }   from '~/model/networks/efficientnetlite_matting_small'
import { RenderOp } from '~/model/render_op'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import type { Backend, Dtype, Tensor } from '~/model/backend'
import type { ModelWeights } from '~/model/weights'

const VIDEO_URL    = '/loop_video.mp4'
const BG_IMAGE_URL = '/demo.jpg'

const WARMUP_FRAMES = 5

interface NetworkLike { readonly output: Tensor; run(): void }
type NetworkCtor = new (backend: Backend, input: Tensor, w: ModelWeights) => NetworkLike

interface BenchPreset {
  name:      string   // label for the results table
  modelFile: string   // weights file basename (without extension), e.g. 'large'
  ctor:      NetworkCtor
  netW:      number
  netH:      number
}

// Mirrors src/pipeline/presets.ts. xs/xxs share Small architecture; medium
// shares Large architecture — only the input resolution and weights file
// differ at the model layer.
const BENCH_PRESETS: BenchPreset[] = [
  { name: 'xl',      modelFile: 'xl',      ctor: EfficientNetLiteMattingXL      as unknown as NetworkCtor, netW: 512, netH: 288 },
  { name: 'large',   modelFile: 'large',   ctor: EfficientNetLiteMattingLarge   as unknown as NetworkCtor, netW: 256, netH: 144 },
  { name: 'medium',  modelFile: 'medium',  ctor: EfficientNetLiteMattingLarge   as unknown as NetworkCtor, netW: 256, netH: 144 },
  { name: 'compact', modelFile: 'compact', ctor: EfficientNetLiteMattingCompact as unknown as NetworkCtor, netW: 256, netH: 144 },
  { name: 'small',   modelFile: 'small',   ctor: EfficientNetLiteMattingSmall   as unknown as NetworkCtor, netW: 256, netH: 144 },
  { name: 'xs',      modelFile: 'xs',      ctor: EfficientNetLiteMattingSmall   as unknown as NetworkCtor, netW: 192, netH: 108 },
  { name: 'xxs',     modelFile: 'xxs',     ctor: EfficientNetLiteMattingSmall   as unknown as NetworkCtor, netW: 128, netH: 72  },
]

const MP_WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
const MP_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'

// ── UI helpers ────────────────────────────────────────────────────────────

const statusEl  = () => document.getElementById('status') as HTMLElement
const resultsEl = () => document.getElementById('results') as HTMLElement

const status = (s: string) => {
  statusEl().textContent = s
  console.log('[bench]', s)
}

function appendRow(label: string, frames: number, seconds: number, fps: number, note = '') {
  const tr = document.createElement('tr')
  tr.innerHTML =
    `<td>${label}</td>` +
    `<td>${frames}</td>` +
    `<td>${seconds.toFixed(2)}s</td>` +
    `<td><b>${fps.toFixed(1)}</b></td>` +
    `<td class="note">${note}</td>`
  resultsEl().appendChild(tr)
}

// ── Asset loading ─────────────────────────────────────────────────────────

async function loadImage(url: string): Promise<ImageBitmap> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`)
  return createImageBitmap(await r.blob())
}

async function loadVideoFile(url: string): Promise<File> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`)
  const buf = await r.arrayBuffer()
  return new File([buf], 'loop_video.mp4', { type: 'video/mp4' })
}

async function cropToAspect(img: ImageBitmap, aspect: number): Promise<ImageBitmap> {
  const srcAspect = img.width / img.height
  let sx: number, sy: number, sw: number, sh: number
  if (srcAspect > aspect) {
    sh = img.height
    sw = Math.round(sh * aspect)
    sx = Math.round((img.width - sw) / 2)
    sy = 0
  } else {
    sw = img.width
    sh = Math.round(sw / aspect)
    sx = 0
    sy = Math.round((img.height - sh) / 2)
  }
  return createImageBitmap(img, sx, sy, sw, sh)
}

// ── Common bench harness ──────────────────────────────────────────────────

interface BenchPipeline {
  // Run forward N times on a synthetic input before the streaming pipeline
  // starts, so shader compile / pipeline cache / weight upload / JIT all
  // happen outside the timed window.
  warmup(src: ImageBitmap, iterations?: number): Promise<void>
  process(frame: VideoFrame): Promise<VideoFrame>
  destroy(): void | Promise<void>
}

function pickEncoderConfig(decoderConfig: VideoDecoderConfig, framerate = 30, bitrate = 4_000_000): VideoEncoderConfig {
  return {
    codec: 'avc1.4d0034',
    width: decoderConfig.codedWidth!,
    height: decoderConfig.codedHeight!,
    framerate,
    bitrate,
  }
}

async function runBench(label: string, pipeline: BenchPipeline, file: File, decoderConfig: VideoDecoderConfig): Promise<{ frames: number, seconds: number, fps: number }> {
  const demuxer = new SimpleDemuxer(file)
  await demuxer.load()

  const muxer = new SimpleMuxer({ video: 'avc' })
  const encoderConfig = pickEncoderConfig(decoderConfig)

  let warmup = WARMUP_FRAMES
  let frames = 0
  let t0 = 0

  status(`${label}: streaming…`)

  await demuxer.videoStream()
    .pipeThrough(new VideoDecodeStream(decoderConfig))
    .pipeThrough(new VideoProcessStream(async (frame: VideoFrame) => {
      if (warmup > 0) {
        warmup--
        if (warmup === 0) t0 = performance.now()
      } else {
        frames++
      }
      const out = await pipeline.process(frame)
      frame.close()
      return out
    }))
    .pipeThrough(new VideoEncodeStream(encoderConfig))
    .pipeTo(muxer.videoSink())

  const seconds = (performance.now() - t0) / 1000
  const fps = frames / seconds
  return { frames, seconds, fps }
}

// ── Longpipe pipeline ─────────────────────────────────────────────────────

async function fetchWeights(modelFile: string, dtype: Dtype): Promise<ArrayBuffer> {
  const tryUrl = async (url: string) => {
    const r = await fetch(url)
    if (!r.ok) return null
    const ct = r.headers.get('content-type') ?? ''
    if (ct.startsWith('text/html')) return null   // vite SPA fallback
    return r.arrayBuffer()
  }
  if (dtype === 'f16') {
    const f16 = await tryUrl(`/model_${modelFile}.f16.bin`)
    if (f16) return f16
  }
  const f32 = await tryUrl(`/model_${modelFile}.bin`)
  if (!f32) throw new Error(`failed to fetch /model_${modelFile}.bin`)
  return f32
}

class LongpipeBench implements BenchPipeline {
  static async create(canvas: HTMLCanvasElement, bgImg: ImageBitmap, preset: BenchPreset): Promise<LongpipeBench> {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    const dtype: Dtype = (await WebGPUBackend.hasF16Support()) ? 'f16' : 'f32'
    const backend = await WebGPUBackend.create({ canvas, dtype })

    const weightsBuf = await fetchWeights(preset.modelFile, dtype)
    const weights = loadWeightsFromBinary(weightsBuf) as ModelWeights

    const networkInput = backend.ops.Input(preset.netH, preset.netW)
    const network = new preset.ctor(backend, networkInput.output, weights)

    const targetAspect = canvas.width / canvas.height
    const bgCropped = await cropToAspect(bgImg, targetAspect)
    const bgInput = backend.ops.Input(canvas.height, canvas.width)
    bgInput.setSource(bgCropped)
    bgInput.run()

    const renderOp = new RenderOp(backend)
    renderOp.attachNetwork(network, networkInput, {
      upscaler: 'bilinear',
      background: { mode: 'image', image: bgInput.output },
    })

    return new LongpipeBench(backend, canvas, renderOp, dtype)
  }

  private constructor(
    private backend: Backend,
    private canvas: HTMLCanvasElement,
    private renderOp: RenderOp,
    public readonly dtype: Dtype,
  ) {}

  async warmup(src: ImageBitmap, iterations = 2): Promise<void> {
    for (let i = 0; i < iterations; i++) {
      this.renderOp.setSource(src)
      this.renderOp.runModel()
      this.renderOp.runDisplay()
    }
    await this.backend.sync()
  }

  async process(frame: VideoFrame): Promise<VideoFrame> {
    this.renderOp.setSource(frame)
    this.renderOp.runModel()
    this.renderOp.runDisplay()
    return new VideoFrame(this.canvas, { timestamp: frame.timestamp, duration: frame.duration ?? undefined })
  }

  destroy() { this.backend.destroy() }
}

// ── MediaPipe pipeline (bare WebGPU compositor) ───────────────────────────

const MP_WGSL = /* wgsl */ `
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  // Fullscreen triangle.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  var out: VsOut;
  let xy = p[vi];
  out.pos = vec4<f32>(xy, 0.0, 1.0);
  // UV: y flipped so (0,0) is top-left like the source frame.
  out.uv  = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
  return out;
}

struct U {
  maskW: u32,
  maskH: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var samp:    sampler;
@group(0) @binding(1) var frameTex: texture_external;
@group(0) @binding(2) var bgTex:    texture_2d<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

fn sampleMask(uv: vec2<f32>) -> f32 {
  let mw = f32(u.maskW);
  let mh = f32(u.maskH);
  let xf = clamp(uv.x * mw - 0.5, 0.0, mw - 1.0);
  let yf = clamp(uv.y * mh - 0.5, 0.0, mh - 1.0);
  let x0 = i32(floor(xf));
  let y0 = i32(floor(yf));
  let x1 = min(x0 + 1, i32(u.maskW) - 1);
  let y1 = min(y0 + 1, i32(u.maskH) - 1);
  let tx = xf - f32(x0);
  let ty = yf - f32(y0);
  let m00 = mask[u32(y0) * u.maskW + u32(x0)];
  let m01 = mask[u32(y0) * u.maskW + u32(x1)];
  let m10 = mask[u32(y1) * u.maskW + u32(x0)];
  let m11 = mask[u32(y1) * u.maskW + u32(x1)];
  return mix(mix(m00, m01, tx), mix(m10, m11, tx), ty);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let fg = textureSampleBaseClampToEdge(frameTex, samp, in.uv);
  let bg = textureSample(bgTex, samp, in.uv);
  let m  = sampleMask(in.uv);
  return vec4<f32>(mix(bg.rgb, fg.rgb, m), 1.0);
}
`

class MediaPipeBench implements BenchPipeline {
  static async create(canvas: HTMLCanvasElement, bgImg: ImageBitmap): Promise<MediaPipeBench> {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('no WebGPU adapter')
    const device = await adapter.requestDevice()

    const ctx = canvas.getContext('webgpu') as GPUCanvasContext
    const format = navigator.gpu.getPreferredCanvasFormat()
    ctx.configure({ device, format, alphaMode: 'opaque' })

    // Upload bg image (cropped to canvas aspect)
    const bgCropped = await cropToAspect(bgImg, canvas.width / canvas.height)
    const bgTex = device.createTexture({
      size: { width: canvas.width, height: canvas.height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture(
      { source: bgCropped },
      { texture: bgTex },
      { width: canvas.width, height: canvas.height },
    )

    // MediaPipe returns the confidence mask at the *input frame resolution*,
    // not the model's native 256×144 — the upsample happens inside the
    // segmenter and the user pays the cost on the way back to GPU.
    const MASK_W = canvas.width
    const MASK_H = canvas.height
    const maskBuf = device.createBuffer({
      size: MASK_W * MASK_H * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const uniformBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([MASK_W, MASK_H, 0, 0]))

    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })

    const module = device.createShaderModule({ code: MP_WGSL })
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    // MediaPipe segmenter
    const tv = await import(/* @vite-ignore */ 'https://esm.sh/@mediapipe/tasks-vision') as any
    const fileset = await tv.FilesetResolver.forVisionTasks(MP_WASM_URL)
    let usedDelegate = 'GPU'
    let segmenter: any
    try {
      segmenter = await tv.ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MP_MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      })
    } catch (e) {
      console.warn('[bench] MediaPipe GPU delegate failed, falling back to CPU:', e)
      usedDelegate = 'CPU'
      segmenter = await tv.ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MP_MODEL_URL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      })
    }

    return new MediaPipeBench(device, ctx, pipeline, sampler, bgTex, maskBuf, uniformBuf, segmenter, MASK_W, MASK_H, usedDelegate)
  }

  private constructor(
    private device: GPUDevice,
    private ctx: GPUCanvasContext,
    private pipeline: GPURenderPipeline,
    private sampler: GPUSampler,
    private bgTex: GPUTexture,
    private maskBuf: GPUBuffer,
    private uniformBuf: GPUBuffer,
    private segmenter: any,
    private maskW: number,
    private maskH: number,
    public readonly delegate: string,
  ) {}

  async warmup(src: ImageBitmap, iterations = 2): Promise<void> {
    // Wrap as a VideoFrame so segmentForVideo + importExternalTexture are
    // both exercised on the same code path the streaming pipeline will use.
    for (let i = 0; i < iterations; i++) {
      const vf = new VideoFrame(src, { timestamp: i * 33333 })
      const out = await this.process(vf)
      out.close()
      vf.close()
    }
    await this.device.queue.onSubmittedWorkDone()
  }

  async process(frame: VideoFrame): Promise<VideoFrame> {
    // Run MediaPipe — synchronous return on web; callback variant accepted too.
    const result: any = this.segmenter.segmentForVideo(frame, performance.now())
    const conf = result.confidenceMasks?.[0] ?? result.confidence_masks?.[0]
    const float32: Float32Array = conf.getAsFloat32Array()
    this.device.queue.writeBuffer(this.maskBuf, 0, float32.buffer, float32.byteOffset, float32.byteLength)
    conf.close?.()
    result.close?.()

    // Import the frame as an external GPU texture (zero-copy on most paths).
    const frameTex = this.device.importExternalTexture({ source: frame })

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: frameTex },
        { binding: 2, resource: this.bgTex.createView() },
        { binding: 3, resource: { buffer: this.maskBuf } },
        { binding: 4, resource: { buffer: this.uniformBuf } },
      ],
    })

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.draw(3, 1, 0, 0)
    pass.end()
    this.device.queue.submit([encoder.finish()])

    return new VideoFrame((this.ctx.canvas as HTMLCanvasElement), {
      timestamp: frame.timestamp,
      duration: frame.duration ?? undefined,
    })
  }

  destroy() {
    this.segmenter.close?.()
    this.device.destroy?.()
  }
}

// ── Entry ─────────────────────────────────────────────────────────────────

function newCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  c.style.maxWidth = '480px'
  c.style.height = 'auto'
  c.style.border = '1px solid #ccc'
  document.getElementById('canvases')?.appendChild(c)
  return c
}

async function main() {
  status('loading assets…')
  const [bgImg, file] = await Promise.all([loadImage(BG_IMAGE_URL), loadVideoFile(VIDEO_URL)])

  const probeDemuxer = new SimpleDemuxer(file)
  await probeDemuxer.load()
  const decoderConfig: VideoDecoderConfig = await probeDemuxer.getVideoDecoderConfig()
  const W = decoderConfig.codedWidth!
  const H = decoderConfig.codedHeight!
  status(`video ${W}×${H}`)

  // Longpipe — every preset
  for (const preset of BENCH_PRESETS) {
    status(`initializing Longpipe (${preset.name})…`)
    const canvas = newCanvas(W, H)
    const bench = await LongpipeBench.create(canvas, bgImg, preset)
    status(`warming up Longpipe (${preset.name})…`)
    await bench.warmup(bgImg)
    const file = await loadVideoFile(VIDEO_URL)
    const result = await runBench(`Longpipe (${preset.name})`, bench, file, decoderConfig)
    appendRow(`Longpipe (${preset.name}, ${bench.dtype})`, result.frames, result.seconds, result.fps,
              `${preset.netW}×${preset.netH} input`)
    await bench.destroy()
  }

  // MediaPipe
  status('initializing MediaPipe…')
  const mpCanvas = newCanvas(W, H)
  const mpBench = await MediaPipeBench.create(mpCanvas, bgImg)
  status('warming up MediaPipe…')
  await mpBench.warmup(bgImg)
  const mpFile = await loadVideoFile(VIDEO_URL)
  const mpResult = await runBench('MediaPipe', mpBench, mpFile, decoderConfig)
  appendRow(`MediaPipe (${mpBench.delegate} delegate)`, mpResult.frames, mpResult.seconds, mpResult.fps)
  mpBench.destroy()

  status('done')
}

document.getElementById('run')?.addEventListener('click', () => {
  resultsEl().innerHTML = ''
  document.getElementById('canvases')!.innerHTML = ''
  main().catch(e => {
    console.error(e)
    status(`error: ${e?.message ?? e}`)
  })
})
