import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import addSrc from '~/model/backends/webgl/shaders/add.glsl'

export class AddWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = addSrc

  constructor(backend: WebGLBackend, a: Tensor, b: Tensor) {
    super(backend)

    const ta = a as WebGLTensor

    const outTexture = this.makeTexture(null, ta.texW, ta.texH)
    this.output = { h: a.h, w: a.w, c: a.c, texture: outTexture, texW: ta.texW, texH: ta.texH }

    this.inputs = [a, b]

    this.samplers = [
      { name: 'u_input_a', texture: (a as WebGLTensor).texture },
      { name: 'u_input_b', texture: (b as WebGLTensor).texture },
    ]

    this.defaultSetup()
    this.dispatch = [ta.texW, ta.texH]
  }
}
