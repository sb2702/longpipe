import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/reframe.glsl'

// Apply the view rect to a tensor (same shape in/out); identity while the rect
// is uninitialised.
export class ReframeWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, src: Tensor, rect: Tensor) {
    super(backend)
    const outTexture = this.makeTexture(null, src.w, src.h)
    this.output = { h: src.h, w: src.w, c: 4, texture: outTexture, texW: src.w, texH: src.h }
    this.inputs = [src, rect]

    this.samplers = [
      { name: 'u_src',  texture: (src  as WebGLTensor).texture },
      { name: 'u_rect', texture: (rect as WebGLTensor).texture },
    ]
    this.uniformInts = { u_h: src.h, u_w: src.w }
    this.defaultSetup()
    this.dispatch = [src.w, src.h]
  }
}
