import type { Tensor, MLBuffer, DownAdapterParams } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import downAdapterSrc from '~/model/backends/webgl/shaders/down_adapter.glsl'
import { convOutSize, resolvePad } from '~/model/backends/webgpu/ops/conv_utils.ts'
import { toUploadView, padToVec4 } from '~/utils/weights.ts'

// Fused stride-N 3×3 conv (4→4) + relu + 1×1 adapter (4→3). Output vec4(xyz, 0).
export class DownAdapterWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = downAdapterSrc

  constructor(backend: WebGLBackend, input: Tensor, downW: Conv2DWeights, adaptW: Conv2DWeights, params: DownAdapterParams) {
    super(backend)

    const outH = convOutSize(input.h, 3, params.stride, 1)
    const outW = convOutSize(input.w, 3, params.stride, 1)
    const pad  = resolvePad(1, input.h, outH, 3, params.stride)  // symmetric → same for w

    // down_w: 9 mat4x4 = 144 floats = 36 vec4. adapt_w: 1 mat4x4 = 4 vec4.
    const downWTex  = this.makeTexture(toUploadView(downW.weights), 36, 1)
    const downBTex  = this.makeTexture(padToVec4(downW.bias), 1, 1)
    const adaptWTex = this.makeTexture(toUploadView(adaptW.weights), 4, 1)
    const adaptBTex = this.makeTexture(padToVec4(adaptW.bias), 1, 1)

    const outTexW    = outW  // 1 output group → texW == outW
    const outTexture = this.makeTexture(null, outTexW, outH)
    this.output = { h: outH, w: outW, c: 4, texture: outTexture, texW: outTexW, texH: outH }
    this.inputs  = [input]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_down_w',  texture: downWTex },
      { name: 'u_down_b',  texture: downBTex },
      { name: 'u_adapt_w', texture: adaptWTex },
      { name: 'u_adapt_b', texture: adaptBTex },
    ]

    this.uniformInts = {
      u_in_w:  input.w,
      u_in_h:  input.h,
      u_stride: params.stride,
      u_pad:    pad,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outH]
  }
}
