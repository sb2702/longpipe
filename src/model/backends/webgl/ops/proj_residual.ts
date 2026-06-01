import type { Tensor, MLBuffer, ProjResidualParams } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import projResidualSrc from '~/model/backends/webgl/shaders/proj_residual.glsl'
import { toUploadView } from '~/utils/weights.ts'

// Bespoke 1×1 conv (no activation) + residual add. `input` is the depthwise
// output (mid channels); `skip` is the residual at out channels.
export class ProjResidualWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = projResidualSrc

  constructor(backend: WebGLBackend, input: Tensor, skip: Tensor, w: Conv2DWeights, params: ProjResidualParams) {
    super(backend)

    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (inGroups*4, outGroups) — K=1, so no K*K rows.
    const weightTex = this.makeTexture(weightData, inGroups * 4, outGroups)
    const biasTex   = this.makeTexture(biasData, outGroups, 1)

    const outTexW    = input.w * outGroups
    const outTexture = this.makeTexture(null, outTexW, input.h)
    this.output = { h: input.h, w: input.w, c: params.outChannels, texture: outTexture, texW: outTexW, texH: input.h }
    this.inputs  = [input, skip]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_skip',    texture: (skip  as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = {
      u_in_c_groups:  inGroups,
      u_out_c_groups: outGroups,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, input.h]
  }
}
