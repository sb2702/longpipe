import type { Tensor } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import concatSrc from '~/model/backends/webgl/shaders/channel_concat.glsl'

export class ChannelConcatWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = concatSrc

  constructor(backend: WebGLBackend, a: Tensor, b: Tensor) {
    super(backend)

    const aGroups   = a.c / 4
    const bGroups   = b.c / 4
    const outGroups = aGroups + bGroups
    const outC      = a.c + b.c
    const outTexW   = a.w * outGroups

    const outTexture = this.makeTexture(null, outTexW, a.h)
    this.output = { h: a.h, w: a.w, c: outC, texture: outTexture, texW: outTexW, texH: a.h }
    this.inputs = [a, b]

    this.samplers = [
      { name: 'u_input_a', texture: (a as WebGLTensor).texture },
      { name: 'u_input_b', texture: (b as WebGLTensor).texture },
    ]

    this.uniformInts = {
      u_a_c_groups: aGroups,
      u_b_c_groups: bGroups,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, a.h]
  }
}
