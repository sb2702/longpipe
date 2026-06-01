import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import catConv6to2Src from '~/model/backends/webgl/shaders/cat_conv_6to2.glsl'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// Fused concat(u, d) + 6→2 conv 3×3 + relu (E up1_combine). `u` = c_up=2 carrier
// (.xy); `d` = c_high=4 skip (full vec4). Output = c_up=2 carrier (.xy).
export class CatConv6to2WebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = catConv6to2Src

  constructor(backend: WebGLBackend, u: Tensor, d: Tensor, w: Conv2DWeights) {
    super(backend)

    // Weight texture: 9 * 2 mat3x2 (6 floats each) = 108 floats = 27 vec4 texels.
    const weightTex = this.makeTexture(toUploadView(w.weights), 27, 1)
    const biasTex   = this.makeTexture(padToVec4(w.bias), 1, 1)

    const outTexW    = u.w  // 1 output group → texW == w
    const outTexture = this.makeTexture(null, outTexW, u.h)
    this.output = { h: u.h, w: u.w, c: 4, texture: outTexture, texW: outTexW, texH: u.h }
    this.inputs  = [u, d]
    this.weights = []

    this.samplers = [
      { name: 'u_u_in',    texture: (u as WebGLTensor).texture },
      { name: 'u_d_in',    texture: (d as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = { u_w: u.w, u_h: u.h }

    this.defaultSetup()
    this.dispatch = [outTexW, u.h]
  }
}
