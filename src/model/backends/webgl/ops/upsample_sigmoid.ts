import type { Tensor, UpsampleParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import upsampleSigmoidSrc from '~/model/backends/webgl/shaders/upsample_sigmoid.glsl'

export class UpsampleSigmoidWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = upsampleSigmoidSrc

  constructor(backend: WebGLBackend, input: Tensor, params: UpsampleParams) {
    super(backend)

    const cGroups    = input.c / 4
    const outTexW    = params.outW * cGroups
    const outTexture = this.makeTexture(null, outTexW, params.outH)

    this.output = { h: params.outH, w: params.outW, c: input.c, texture: outTexture, texW: outTexW, texH: params.outH }
    this.inputs = [input]

    this.samplers = [
      { name: 'u_input', texture: (input as WebGLTensor).texture },
    ]

    this.uniformInts = {
      u_in_w:     input.w,
      u_in_h:     input.h,
      u_out_w:    params.outW,
      u_out_h:    params.outH,
      u_c_groups: cGroups,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, params.outH]
  }
}
