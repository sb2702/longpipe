import type { Tensor, UpsampleParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import upsampleConcatSrc from '~/model/backends/webgl/shaders/upsample_concat.glsl'

export class UpsampleConcatWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = upsampleConcatSrc

  constructor(backend: WebGLBackend, a: Tensor, b: Tensor, params: UpsampleParams) {
    super(backend)

    const aGroups   = a.c / 4
    const bGroups   = b.c / 4
    const outGroups = aGroups + bGroups
    const outC      = a.c + b.c
    const outTexW   = params.outW * outGroups

    const outTexture = this.makeTexture(null, outTexW, params.outH)
    this.output = { h: params.outH, w: params.outW, c: outC, texture: outTexture, texW: outTexW, texH: params.outH }
    this.inputs = [a, b]

    this.samplers = [
      { name: 'u_input_a', texture: (a as WebGLTensor).texture },
      { name: 'u_input_b', texture: (b as WebGLTensor).texture },
    ]

    this.uniformInts = {
      u_in_w:       a.w,
      u_in_h:       a.h,
      u_out_w:      params.outW,
      u_out_h:      params.outH,
      u_a_c_groups: aGroups,
      u_b_c_groups: bGroups,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, params.outH]
  }
}
