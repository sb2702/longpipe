import type { Tensor, MLBuffer } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import gatesSrc from '~/model/backends/webgl/shaders/gates_fused.glsl'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// ConvGRU z + r gates, fused (production config c_up=2, recurrent=1).
export class GatesFusedWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = gatesSrc

  constructor(backend: WebGLBackend, uIn: Tensor, hPrev: Tensor, w: Conv2DWeights) {
    super(backend)

    // Weight texture: (9, 1) — one vec4 per kpos. Bias texture: (1, 1).
    const weightTex = this.makeTexture(toUploadView(w.weights), 9, 1)
    const biasTex   = this.makeTexture(padToVec4(w.bias), 1, 1)

    const outTexW    = uIn.w  // 1 output group → texW == w
    const outTexture = this.makeTexture(null, outTexW, uIn.h)
    this.output = { h: uIn.h, w: uIn.w, c: 4, texture: outTexture, texW: outTexW, texH: uIn.h }
    this.inputs  = [uIn, hPrev]
    this.weights = []

    this.samplers = [
      { name: 'u_u_in',   texture: (uIn   as WebGLTensor).texture },
      { name: 'u_h_prev', texture: (hPrev as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = { u_w: uIn.w, u_h: uIn.h }

    this.defaultSetup()
    this.dispatch = [outTexW, uIn.h]
  }
}
