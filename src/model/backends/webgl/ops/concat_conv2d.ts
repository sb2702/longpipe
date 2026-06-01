import type { Tensor, MLBuffer, ConcatConv2dParams } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import concatConv2dSrc from '~/model/backends/webgl/shaders/concat_conv2d.glsl'
import { toUploadView } from '~/utils/weights.ts'

// Fused concat(a, b) → 3×3 conv (pad 1) → relu6. `a` and `b` must share the
// same spatial resolution; the conv weight's input channels are ordered [a, b].
export class ConcatConv2dWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = concatConv2dSrc

  constructor(backend: WebGLBackend, a: Tensor, b: Tensor, w: Conv2DWeights, params: ConcatConv2dParams) {
    super(backend)

    const aGroups   = a.c / 4
    const bGroups   = b.c / 4
    const inGroups  = aGroups + bGroups
    const outGroups = params.outChannels / 4

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (inGroups*4, 9*outGroups) — kernel 3×3 → 9 kpos rows per
    // output group. in_groups = a_groups + b_groups, channels ordered [a, b].
    const weightTex = this.makeTexture(weightData, inGroups * 4, 9 * outGroups)
    const biasTex   = this.makeTexture(biasData, outGroups, 1)

    const outTexW    = a.w * outGroups
    const outTexture = this.makeTexture(null, outTexW, a.h)
    this.output = { h: a.h, w: a.w, c: params.outChannels, texture: outTexture, texW: outTexW, texH: a.h }
    this.inputs  = [a, b]
    this.weights = []

    this.samplers = [
      { name: 'u_a',       texture: (a as WebGLTensor).texture },
      { name: 'u_b',       texture: (b as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = {
      u_w:            a.w,
      u_h:            a.h,
      u_a_groups:     aGroups,
      u_b_groups:     bGroups,
      u_out_c_groups: outGroups,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, a.h]
  }
}
