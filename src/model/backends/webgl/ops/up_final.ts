import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import upFinalSrc from '~/model/backends/webgl/shaders/up_final.glsl'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// A/B alpha head: fused concat(u, rgb) → conv 3×3 5→1 → sigmoid.
export class UpFinalWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = upFinalSrc

  constructor(backend: WebGLBackend, u: Tensor, rgb: Tensor, w: Conv2DWeights) {
    super(backend)

    const weightTex = this.makeTexture(toUploadView(w.weights), 18, 1)  // 18 vec4
    const biasTex   = this.makeTexture(padToVec4(w.bias), 1, 1)

    const outTexW    = u.w  // 1 output group → texW == w
    const outTexture = this.makeTexture(null, outTexW, u.h)
    this.output = { h: u.h, w: u.w, c: 4, texture: outTexture, texW: outTexW, texH: u.h }
    this.inputs  = [u, rgb]
    this.weights = []

    this.samplers = [
      { name: 'u_u_gru',   texture: (u   as WebGLTensor).texture },
      { name: 'u_rgb',     texture: (rgb as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = { u_w: u.w, u_h: u.h }

    this.defaultSetup()
    this.dispatch = [outTexW, u.h]
  }
}
