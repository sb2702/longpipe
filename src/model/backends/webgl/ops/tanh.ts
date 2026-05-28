import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import tanhSrc from '~/model/backends/webgl/shaders/tanh.glsl'

export class TanhWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = tanhSrc

  constructor(backend: WebGLBackend, input: Tensor) {
    super(backend)

    const ti = input as WebGLTensor

    const outTexture = this.makeTexture(null, ti.texW, ti.texH)
    this.output = { h: input.h, w: input.w, c: input.c, texture: outTexture, texW: ti.texW, texH: ti.texH }

    this.inputs = [input]

    this.samplers = [
      { name: 'u_input', texture: ti.texture },
    ]

    this.defaultSetup()
    this.dispatch = [ti.texW, ti.texH]
  }
}
