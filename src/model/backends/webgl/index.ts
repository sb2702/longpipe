import type { Backend } from '~/model/backend'
import type { WebGLTensor, WebGLMLBuffer } from '~/model/backends/webgl/base_webgl_op'
import { Conv2DWebGL } from '~/model/backends/webgl/ops/conv2d'
import { DepthwiseConv2DWebGL } from '~/model/backends/webgl/ops/depthwise_conv2d'
import { AddWebGL } from '~/model/backends/webgl/ops/add'

export class WebGLBackend implements Backend {
  readonly ops: Backend['ops']
  readonly fbo: WebGLFramebuffer

  private constructor(readonly gl: WebGL2RenderingContext) {
    this.fbo = gl.createFramebuffer()!
    const notImpl = (): never => { throw new Error('not implemented in WebGL backend') }
    this.ops = {
      Conv2d:           (input, weights, bias, params) => new Conv2DWebGL(this, input, weights, bias, params),
      DepthwiseConv2d:  (input, weights, bias, params) => new DepthwiseConv2DWebGL(this, input, weights, bias, params),
      Add:              (a, b)                         => new AddWebGL(this, a, b),
      Sigmoid:          notImpl,
      BilinearUpsample: notImpl,
      ChannelConcat:    notImpl,
      Conv2dAdd:        notImpl,
      UpsampleConcat:   notImpl,
      UpsampleConv1x1:  notImpl,
      UpsampleSigmoid:  notImpl,
    }
  }

  static create(): WebGLBackend {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2')
    if (!gl) throw new Error('WebGL2 not available')
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float not available')
    return new WebGLBackend(gl)
  }

  tensor(h: number, w: number, c: number, data?: Float32Array): WebGLTensor {
    const texW = w * (c / 4)
    const texH = h
    const gl   = this.gl
    const tex  = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, data ?? null)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return { h, w, c, texture: tex, texW, texH }
  }

  upload(data: Float32Array): WebGLMLBuffer {
    return { data }
  }

  async readback(tensor: WebGLTensor): Promise<Float32Array> {
    const gl     = this.gl
    const pixels = new Float32Array(tensor.texW * tensor.texH * 4)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tensor.texture, 0)
    gl.readPixels(0, 0, tensor.texW, tensor.texH, gl.RGBA, gl.FLOAT, pixels)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    return pixels
  }

  destroy(): void {
    this.gl.deleteFramebuffer(this.fbo)
  }
}
