import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import upFinalSkipSrc from '~/model/backends/webgl/shaders/up_final_skip.glsl'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// C/D alpha head: fused concat(u, d_full, rgb) → conv 3×3 9→1 → sigmoid.
export class UpFinalSkipWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = upFinalSkipSrc

  constructor(backend: WebGLBackend, u: Tensor, dFull: Tensor, rgb: Tensor, w: Conv2DWeights) {
    super(backend)

    const weightTex = this.makeTexture(toUploadView(w.weights), 27, 1)  // 27 vec4
    const biasTex   = this.makeTexture(padToVec4(w.bias), 1, 1)

    const outTexW    = u.w  // 1 output group → texW == w
    const outTexture = this.makeTexture(null, outTexW, u.h)
    this.output = { h: u.h, w: u.w, c: 4, texture: outTexture, texW: outTexW, texH: u.h }
    this.inputs  = [u, dFull, rgb]
    this.weights = []

    this.samplers = [
      { name: 'u_u_gru',   texture: (u     as WebGLTensor).texture },
      { name: 'u_d_full',  texture: (dFull as WebGLTensor).texture },
      { name: 'u_rgb',     texture: (rgb   as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = { u_w: u.w, u_h: u.h }

    this.defaultSetup()
    this.dispatch = [outTexW, u.h]
  }
}
