import type { Backend, Dtype, DataView_ } from '~/model/backend.ts'
import type { WebGLTensor, WebGLMLBuffer } from '~/model/backends/webgl/base_webgl_op.ts'
import { float32ArrayToHalf, halfArrayToFloat32 } from '~/utils/fp16.ts'
import { Conv2DWebGL } from '~/model/backends/webgl/ops/conv2d.ts'
import { DepthwiseConv2DWebGL } from '~/model/backends/webgl/ops/depthwise_conv2d.ts'
import { AddWebGL } from '~/model/backends/webgl/ops/add.ts'
import { SigmoidWebGL } from '~/model/backends/webgl/ops/sigmoid.ts'
import { TanhWebGL } from '~/model/backends/webgl/ops/tanh.ts'
import { ElementwiseMulWebGL } from '~/model/backends/webgl/ops/elementwise_mul.ts'
import { BilinearUpsampleWebGL } from '~/model/backends/webgl/ops/bilinear_upsample.ts'
import { BicubicUpsampleWebGL  } from '~/model/backends/webgl/ops/bicubic_upsample.ts'
import { ChannelConcatWebGL } from '~/model/backends/webgl/ops/channel_concat.ts'
import { UpsampleSigmoidWebGL } from '~/model/backends/webgl/ops/upsample_sigmoid.ts'
import { UpsampleConcatWebGL } from '~/model/backends/webgl/ops/upsample_concat.ts'
import { UpsampleConv1x1WebGL } from '~/model/backends/webgl/ops/upsample_conv1x1.ts'
import { Conv2dAddWebGL } from '~/model/backends/webgl/ops/conv2d_add.ts'
import { ProjResidualWebGL } from '~/model/backends/webgl/ops/proj_residual.ts'
import { ConcatConv2dWebGL } from '~/model/backends/webgl/ops/concat_conv2d.ts'
import { GatesFusedWebGL } from '~/model/backends/webgl/ops/gates_fused.ts'
import { CandUpdateFusedWebGL } from '~/model/backends/webgl/ops/cand_update_fused.ts'
import { ConvExpandWebGL } from '~/model/backends/webgl/ops/conv_expand.ts'
import { CatConv6to2WebGL } from '~/model/backends/webgl/ops/cat_conv_6to2.ts'
import { DownAdapterWebGL } from '~/model/backends/webgl/ops/down_adapter.ts'
import { UpFinalWebGL } from '~/model/backends/webgl/ops/up_final.ts'
import { UpFinalSkipWebGL } from '~/model/backends/webgl/ops/up_final_skip.ts'
import { CompositeSolidWebGL } from '~/model/backends/webgl/ops/composite_solid.ts'
import { CompositeImageWebGL } from '~/model/backends/webgl/ops/composite_image.ts'
import { CompositeImageBilinearWebGL } from '~/model/backends/webgl/ops/composite_image_bilinear.ts'
import { CompositePassthroughWebGL } from '~/model/backends/webgl/ops/composite_passthrough.ts'
import { InputWebGL } from '~/model/backends/webgl/ops/input.ts'

export interface WebGLBackendOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  // Defaults to 'f32'. 'f16' switches all activation/weight textures to
  // RGBA16F + HALF_FLOAT — saves bandwidth, but fragment shader compute stays
  // fp32 (GLSL ES 3.00 has no native half type).
  dtype?: Dtype;
}

// Resolved texture format triple: (internalFormat, format, type) for texImage2D
// plus the typed-array constructor expected for upload data.
export interface TextureFormat {
  internalFormat: GLenum  // e.g. RGBA32F or RGBA16F
  format:         GLenum  // RGBA
  type:           GLenum  // FLOAT or HALF_FLOAT
  bytesPerElement: 2 | 4
}

export class WebGLBackend implements Backend {
  readonly ops: Backend['ops']
  readonly presenters: Backend['presenters']
  readonly fbo: WebGLFramebuffer
  readonly textureFormat: TextureFormat

  private constructor(
    readonly gl: WebGL2RenderingContext,
    readonly canvas: HTMLCanvasElement | OffscreenCanvas,
    readonly dtype: Dtype,
  ) {
    this.fbo = gl.createFramebuffer()!
    this.textureFormat = dtype === 'f16'
      ? { internalFormat: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT, bytesPerElement: 2 }
      : { internalFormat: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,      bytesPerElement: 4 }

    this.ops = {
      Conv2d:           (input, weights, params) => new Conv2DWebGL(this, input, weights, params),
      DepthwiseConv2d:  (input, weights, params) => new DepthwiseConv2DWebGL(this, input, weights, params),
      Add:              (a, b)                         => new AddWebGL(this, a, b),
      Sigmoid:          (input)                          => new SigmoidWebGL(this, input),
      Tanh:             (input)                          => new TanhWebGL(this, input),
      ElementwiseMul:   (a, b)                          => new ElementwiseMulWebGL(this, a, b),
      BilinearUpsample: (input, params)                => new BilinearUpsampleWebGL(this, input, params),
      BicubicUpsample:  (input, params)                => new BicubicUpsampleWebGL(this, input, params),
      ChannelConcat:    (a, b)                           => new ChannelConcatWebGL(this, a, b),
      Conv2dAdd:        (input, skip, weights, params)   => new Conv2dAddWebGL(this, input, skip, weights, params),
      ProjResidual:     (input, skip, weights, params)   => new ProjResidualWebGL(this, input, skip, weights, params),
      ConcatConv2d:     (a, b, weights, params)          => new ConcatConv2dWebGL(this, a, b, weights, params),
      GatesFused:       (uIn, hPrev, weights)            => new GatesFusedWebGL(this, uIn, hPrev, weights),
      CandUpdateFused:  (uIn, hPrev, gatesOut, w, gamma) => new CandUpdateFusedWebGL(this, uIn, hPrev, gatesOut, w, gamma),
      ConvExpand:       (input, weights)                 => new ConvExpandWebGL(this, input, weights),
      CatConv6to2:      (u, d, weights)                  => new CatConv6to2WebGL(this, u, d, weights),
      DownAdapter:      (input, downW, adaptW, params)   => new DownAdapterWebGL(this, input, downW, adaptW, params),
      UpFinal:          (u, rgb, weights)                => new UpFinalWebGL(this, u, rgb, weights),
      UpFinalSkip:      (u, dFull, rgb, weights)         => new UpFinalSkipWebGL(this, u, dFull, rgb, weights),
      UpsampleConcat:   (a, b, params)                   => new UpsampleConcatWebGL(this, a, b, params),
      UpsampleConv1x1:  (input, weights, params)         => new UpsampleConv1x1WebGL(this, input, weights, params),
      UpsampleSigmoid:  (input, params)                  => new UpsampleSigmoidWebGL(this, input, params),
      Input:            (h, w)                           => new InputWebGL(this, h, w),
    }
    this.presenters = {
      // WebGL writes to the implicit default framebuffer — the op binds it
      // itself via backend.bindDisplayFramebuffer(), so the wrapper is a
      // straight pass-through.
      CompositeSolid: (image, alpha, bgColor) =>
        new CompositeSolidWebGL(this, image, alpha, bgColor),
      CompositeImage: (image, alpha, bg) =>
        new CompositeImageWebGL(this, image, alpha, bg),
      CompositeImageBilinear: (image, alpha, bg) =>
        new CompositeImageBilinearWebGL(this, image, alpha, bg),
      CompositePassthrough: (image) =>
        new CompositePassthroughWebGL(this, image),
    }
  }

  // Cheap capability probe — works in both Window and Worker scope
  // (OffscreenCanvas is universal on our targets). Probes WebGL2 plus the
  // float-render extension we depend on. The actual create() may still
  // fail on a real canvas due to browser quirks, so callers must handle
  // create() throwing too.
  //
  // We explicitly tear down the probe context via WEBGL_lose_context.
  // Browsers cap simultaneous WebGL contexts (~16 in Chrome); if we leak
  // probe contexts into GC limbo, future production context creation can
  // fail in confusing ways.
  static isAvailable(): boolean {
    let gl: WebGL2RenderingContext | null = null
    try {
      const probe = new OffscreenCanvas(1, 1)
      gl = probe.getContext('webgl2') as WebGL2RenderingContext | null
      if (!gl) return false
      if (!gl.getExtension('EXT_color_buffer_float')) return false
      return true
    } catch {
      return false
    } finally {
      // WEBGL_lose_context is broadly supported. If it isn't here for some
      // reason, we proceed without — the context will be reclaimed by GC
      // eventually, just less promptly.
      gl?.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }

  static create(opts: WebGLBackendOptions): WebGLBackend {
    const gl = opts.canvas.getContext('webgl2') as WebGL2RenderingContext | null
    if (!gl) throw new Error('WebGL2 not available')
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float not available')
    return new WebGLBackend(gl, opts.canvas, opts.dtype ?? 'f32')
  }

  // Bind the canvas (default framebuffer) as the render target. Compositor
  // calls this before its draw to render directly to the display.
  bindDisplayFramebuffer(): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null)
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height)
  }

  // Convert source data to whichever typed array texImage2D expects for the
  // current dtype. Float32 source + f16 dtype gets converted to fp16 bits;
  // Uint16 source + f32 dtype is decoded back to floats. Pass-through if the
  // source already matches.
  toTextureView(data: DataView_ | null): Float32Array | Uint16Array | null {
    if (data === null) return null
    const wantHalf = this.dtype === 'f16'
    const isHalf = data instanceof Uint16Array
    if (wantHalf === isHalf) return data
    return wantHalf
      ? float32ArrayToHalf(data as Float32Array)
      : halfArrayToFloat32(data as Uint16Array)
  }

  tensor(h: number, w: number, c: number, data?: DataView_): WebGLTensor {
    const texW = w * (c / 4)
    const texH = h
    const gl   = this.gl
    const tex  = gl.createTexture()!
    const fmt  = this.textureFormat
    const view = this.toTextureView(data ?? null)
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internalFormat, texW, texH, 0, fmt.format, fmt.type, view)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return { h, w, c, texture: tex, texW, texH }
  }

  // Stash raw weight data — no conversion yet. The op decides when to upload
  // (typically right inside its constructor via base_webgl_op.makeTexture).
  upload(data: DataView_): WebGLMLBuffer {
    return { data }
  }

  async readback(tensor: WebGLTensor): Promise<Float32Array> {
    const gl  = this.gl
    const fmt = this.textureFormat
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tensor.texture, 0)
    if (this.dtype === 'f16') {
      const px = new Uint16Array(tensor.texW * tensor.texH * 4)
      gl.readPixels(0, 0, tensor.texW, tensor.texH, fmt.format, gl.HALF_FLOAT, px)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      return halfArrayToFloat32(px)
    }
    const px = new Float32Array(tensor.texW * tensor.texH * 4)
    gl.readPixels(0, 0, tensor.texW, tensor.texH, fmt.format, gl.FLOAT, px)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    return px
  }

  // GPU-resident texture→texture copy (no CPU round-trip). Binds src as the
  // reusable fbo's color attachment (read source) and copies its pixels into
  // dst with copyTexSubImage2D. src/dst must share texture dimensions + format
  // (guaranteed when same shape + same backend dtype).
  copyTensor(src: WebGLTensor, dst: WebGLTensor): void {
    if (src.texW !== dst.texW || src.texH !== dst.texH)
      throw new Error(`copyTensor: size mismatch (src ${src.texW}×${src.texH} vs dst ${dst.texW}×${dst.texH})`)
    const gl = this.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, src.texture, 0)
    gl.bindTexture(gl.TEXTURE_2D, dst.texture)
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, src.texW, src.texH)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
  }

  // Wait for all pending GL work to complete. Uses fenceSync + polling
  // clientWaitSync (non-blocking, yields to the event loop). Falls back to
  // gl.finish() if fences aren't available.
  async sync(): Promise<void> {
    const gl = this.gl
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
    if (!fence) {
      gl.finish()
      return
    }
    gl.flush()   // ensure commands are submitted, otherwise the fence may never signal
    while (true) {
      const status = gl.clientWaitSync(fence, 0, 0)
      if (status === gl.ALREADY_SIGNALED || status === gl.CONDITION_SATISFIED) {
        gl.deleteSync(fence)
        return
      }
      if (status === gl.WAIT_FAILED) {
        gl.deleteSync(fence)
        throw new Error('WebGLBackend.sync: clientWaitSync returned WAIT_FAILED')
      }
      // Yield ~1ms then poll again. setTimeout granularity caps how tight
      // this loop is; for most preset benches the actual GPU work
      // dominates so this overhead is small.
      await new Promise<void>(r => setTimeout(r, 1))
    }
  }

  destroy(): void {
    // WEBGL_lose_context.loseContext() releases the entire context and
    // all resources allocated from it (textures, programs, buffers,
    // framebuffers) in one shot — far simpler and more thorough than
    // tracking and individually deleting each object.
    //
    // Browsers cap simultaneous WebGL contexts (~16 in Chrome); failing
    // to release here means a long-lived SPA that creates/destroys
    // multiple Pipelines will eventually fail to create new contexts.
    //
    // Extension is broadly supported but not universal; if missing, the
    // context will be reclaimed by GC eventually (just less promptly).
    this.gl.deleteFramebuffer(this.fbo)
    this.gl.getExtension('WEBGL_lose_context')?.loseContext()
  }
}
