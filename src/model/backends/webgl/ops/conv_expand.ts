import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import convExpandSrc from '~/model/backends/webgl/shaders/conv_expand.glsl'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// Bespoke N→2 conv 3×3 (pad 1) + relu (wrapper expand_feat). Output carrier
// .xy = 2 native channels, .zw = 0.
export class ConvExpandWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = convExpandSrc

  constructor(backend: WebGLBackend, input: Tensor, w: Conv2DWeights) {
    super(backend)

    const inGroups = input.c / 4

    // Weight texture: 9 * in_groups mat4x2 (8 floats each) = 18*in_groups vec4
    // texels in a single row. Bounded small (feat_ch ≤ 32 → ≤ 144 texels).
    const weightTex = this.makeTexture(toUploadView(w.weights), 18 * inGroups, 1)
    const biasTex   = this.makeTexture(padToVec4(w.bias), 1, 1)

    const outTexW    = input.w  // 1 output group → texW == w
    const outTexture = this.makeTexture(null, outTexW, input.h)
    this.output = { h: input.h, w: input.w, c: 4, texture: outTexture, texW: outTexW, texH: input.h }
    this.inputs  = [input]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = { u_w: input.w, u_h: input.h, u_in_groups: inGroups }

    this.defaultSetup()
    this.dispatch = [outTexW, input.h]
  }
}
