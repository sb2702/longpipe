import type { Tensor, MLBuffer, UpsampleConv1x1Params } from '~/model/backend'
import type { Conv2DWeights } from '~/model/weights'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import upsampleConv1x1Src from '~/model/backends/webgl/shaders/upsample_conv1x1.glsl'
import { toUploadView } from '~/utils/weights'

export class UpsampleConv1x1WebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = upsampleConv1x1Src

  constructor(backend: WebGLBackend, input: Tensor, w: Conv2DWeights, params: UpsampleConv1x1Params) {
    super(backend)

    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (inGroups * 4, outGroups) — K=1, so kernel-row dim collapses.
    const weightTex = this.makeTexture(weightData, inGroups * 4, outGroups)
    // Bias texture: (outGroups, 1)
    const biasTex   = this.makeTexture(biasData, outGroups, 1)

    const outTexW    = params.outW * outGroups
    const outTexture = this.makeTexture(null, outTexW, params.outH)
    this.output = { h: params.outH, w: params.outW, c: params.outChannels, texture: outTexture, texW: outTexW, texH: params.outH }
    this.inputs  = [input]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = {
      u_in_w:         input.w,
      u_in_h:         input.h,
      u_out_w:        params.outW,
      u_out_h:        params.outH,
      u_in_c_groups:  inGroups,
      u_out_c_groups: outGroups,
      u_activation:   params.activation === 'relu6' ? 1 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, params.outH]
  }
}
