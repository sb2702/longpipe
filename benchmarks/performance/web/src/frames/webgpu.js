// From github.com/sb2702/webrtc-hacks-webgpu (segmentation/web/src/frames/webgpu.js,
// commit 2c6528c, MIT, (c) 2026 Sam Bhattacharyya). unchanged.
//
// WebGPU frame I/O: VideoFrame in -> model input tensor, mask -> VideoFrame out.
//
// Used by every pipeline that runs its pre/post-processing on WebGPU, whatever
// runs the model in between (our shaders, LiteRT on WebGPU, LiteRT/MediaPipe
// on WASM). Tensors here are plain array<f32> with a per-pixel `stride`:
// 4 for our vec4-packed tensors, 3 (input) / 1 (mask) for densely packed
// NHWC tensors like LiteRT's.

// Video frames are imported zero-copy as texture_external; anything else
// (ImageBitmap, canvas, <img>) is copied into an rgba8 texture first.
export const FRAME_BINDING = {
  external: { decl: 'texture_external', sample: (uv) => `textureSampleBaseClampToEdge(frame, samp, ${uv})` },
  texture: { decl: 'texture_2d<f32>', sample: (uv) => `textureSampleLevel(frame, samp, ${uv}, 0.0)` },
}

// Stretch the frame to the model's input size (like MediaPipe). Each output
// pixel averages 4 bilinear taps — roughly a 4x4 box filter — so a 720p ->
// 256x256 downscale doesn't alias.
export const preprocessShader = (kind) => /* wgsl */ `
struct P { out_w: u32, out_h: u32, stride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var frame: ${FRAME_BINDING[kind].decl};
@group(0) @binding(3) var samp: sampler;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.out_w || id.y >= p.out_h) { return; }
  let texel = 1.0 / vec2f(f32(p.out_w), f32(p.out_h));
  let uv = (vec2f(id.xy) + 0.5) * texel;
  var rgb = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    let offset = (vec2f(f32(i & 1u), f32(i >> 1u)) - 0.5) * 0.5 * texel;
    rgb += ${FRAME_BINDING[kind].sample('uv + offset')}.rgb;
  }
  rgb *= 0.25;
  let base = (id.y * p.out_w + id.x) * p.stride;
  dst[base] = rgb.r;
  dst[base + 1u] = rgb.g;
  dst[base + 2u] = rgb.b;
  if (p.stride == 4u) { dst[base + 3u] = 0.0; }
}`

// Full-screen pass: mix(color, frame, mask). The mask is a storage buffer, so
// it's upsampled with a hand-written bilinear filter.
export const compositeShader = (kind) => /* wgsl */ `
struct P { color: vec4f, out_size: vec2f, mask_w: u32, mask_h: u32, stride: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var frame: ${FRAME_BINDING[kind].decl};
@group(0) @binding(3) var samp: sampler;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}

fn mask_at(x: u32, y: u32) -> f32 { return mask[(y * p.mask_w + x) * p.stride]; }

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / p.out_size;
  let size = vec2f(f32(p.mask_w), f32(p.mask_h));
  let s = clamp(uv * size - 0.5, vec2f(0.0), size - 1.0);
  let p0 = vec2u(floor(s));
  let p1 = min(p0 + 1u, vec2u(p.mask_w - 1u, p.mask_h - 1u));
  let f = s - floor(s);
  let alpha = mix(mix(mask_at(p0.x, p0.y), mask_at(p1.x, p0.y), f.x),
                  mix(mask_at(p0.x, p1.y), mask_at(p1.x, p1.y), f.x), f.y);
  let fg = ${FRAME_BINDING[kind].sample('uv')}.rgb;
  return vec4f(mix(p.color.rgb, fg, alpha), 1.0);
}`

export const frameSize = (f) => [f.displayWidth ?? f.videoWidth ?? f.width, f.displayHeight ?? f.videoHeight ?? f.height]

const isExternal = (f) =>
  (typeof VideoFrame !== 'undefined' && f instanceof VideoFrame) ||
  (typeof HTMLVideoElement !== 'undefined' && f instanceof HTMLVideoElement)

export class WebGPUFrames {
  constructor(device) {
    this.device = device
    this.canvas = new OffscreenCanvas(1, 1)
    this.context = this.canvas.getContext('webgpu')
    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' })
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.pipelines = new Map()
    this.uniforms = new Map()
    this.color = [0, 177 / 255, 64 / 255, 1]   // green screen
  }

  setColor([r, g, b]) {
    this.color = [r, g, b, 1]
  }

  // The frame as a bindable resource: zero-copy for VideoFrame / <video>, a
  // GPU-side copy for everything else.
  import(frame) {
    const [width, height] = frameSize(frame)
    if (isExternal(frame)) return { kind: 'external', resource: this.device.importExternalTexture({ source: frame }), width, height }
    if (this.frameTexture?.width !== width || this.frameTexture?.height !== height) {
      this.frameTexture?.destroy()
      this.frameTexture = this.device.createTexture({
        size: [width, height], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
    }
    this.device.queue.copyExternalImageToTexture({ source: frame }, { texture: this.frameTexture }, [width, height])
    return { kind: 'texture', resource: this.frameTexture.createView(), width, height }
  }

  pipeline(kind) {
    if (!this.pipelines.has(kind)) {
      const device = this.device
      const composite = device.createShaderModule({ code: compositeShader(kind) })
      this.pipelines.set(kind, {
        preprocess: device.createComputePipeline({
          layout: 'auto', compute: { module: device.createShaderModule({ code: preprocessShader(kind) }), entryPoint: 'main' },
        }),
        composite: device.createRenderPipeline({
          layout: 'auto',
          vertex: { module: composite, entryPoint: 'vs' },
          fragment: { module: composite, entryPoint: 'fs', targets: [{ format: this.format }] },
        }),
      })
    }
    return this.pipelines.get(kind)
  }

  uniform(name, data) {
    let buffer = this.uniforms.get(name)
    if (!buffer) {
      buffer = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      this.uniforms.set(name, buffer)
    }
    this.device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }

  // Resize `src` into the model input buffer `dst` (width x height x stride f32).
  preprocess(pass, src, dst, { width, height, stride }) {
    const pipeline = this.pipeline(src.kind).preprocess
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform('preprocess', new Uint32Array([width, height, stride, 0])) } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: src.resource },
        { binding: 3, resource: this.sampler },
      ],
    }))
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
  }

  // Draw the frame over the background color, keyed by `mask` (width x height
  // x stride f32, foreground probability in the first channel).
  composite(encoder, src, mask, { width, height, stride }) {
    if (this.canvas.width !== src.width || this.canvas.height !== src.height) {
      this.canvas.width = src.width
      this.canvas.height = src.height
    }
    const params = new ArrayBuffer(48)
    new Float32Array(params, 0, 6).set([...this.color, src.width, src.height])
    new Uint32Array(params, 24, 3).set([width, height, stride])

    const pipeline = this.pipeline(src.kind).composite
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform('composite', params) } },
        { binding: 1, resource: { buffer: mask } },
        { binding: 2, resource: src.resource },
        { binding: 3, resource: this.sampler },
      ],
    }))
    pass.draw(3)
    pass.end()
  }

  // Snapshot the canvas as the output frame. Must run in the same task as the
  // submit that drew it.
  output(frame) {
    return new VideoFrame(this.canvas, { timestamp: frame.timestamp, duration: frame.duration ?? undefined })
  }
}
