import type { Tensor, MLBuffer, ConvTranspose2dParams } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/conv_transpose2d.glsl'
import { convTransposeOutSize } from '~/model/backends/webgpu/ops/conv_utils.ts'
import { toUploadView } from '~/utils/weights.ts'

// Gather-form transposed conv. Weight/bias textures use the SAME layout as
// conv2d (the flat weight buffer is identical) — only the spatial mapping in the
// shader differs.
export class ConvTranspose2DWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor, w: Conv2DWeights, params: ConvTranspose2dParams) {
    super(backend)

    const outH      = convTransposeOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW      = convTransposeOutSize(input.w, params.kernel, params.stride, params.padding)
    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (inGroups * 4, kernel² * outGroups) — identical to conv2d.
    const weightTex = this.makeTexture(weightData, inGroups * 4, params.kernel * params.kernel * outGroups)
    const biasTex   = this.makeTexture(biasData, outGroups, 1)

    const outTexW    = outW * outGroups
    const outTexture = this.makeTexture(null, outTexW, outH)
    this.output  = { h: outH, w: outW, c: params.outChannels, texture: outTexture, texW: outTexW, texH: outH }
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
      u_in_c_groups:  inGroups,
      u_out_c_groups: outGroups,
      u_kernel_h:     params.kernel,
      u_kernel_w:     params.kernel,
      u_stride:       params.stride,
      u_pad_top:      params.padding,
      u_pad_left:     params.padding,
      u_activation:   params.activation === 'relu6' ? 1 : params.activation === 'relu' ? 2 : params.activation === 'leaky' ? 3 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outH]
  }
}
