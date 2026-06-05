import type { Tensor, MLBuffer, Conv2dParams } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import conv2dSrc from '~/model/backends/webgl/shaders/conv2d.glsl'
import { convOutSize, resolvePad } from '~/model/backends/webgpu/ops/conv_utils.ts'
import { toUploadView } from '~/utils/weights.ts'

export class Conv2DWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = conv2dSrc

  constructor(backend: WebGLBackend, input: Tensor, w: Conv2DWeights, params: Conv2dParams) {
    super(backend)

    const outH      = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW      = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4
    const padTop    = resolvePad(params.padding, input.h, outH, params.kernel, params.stride)
    const padLeft   = resolvePad(params.padding, input.w, outW, params.kernel, params.stride)

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (inGroups * 4, kernel² * outGroups)
    const weightTex = this.makeTexture(weightData, inGroups * 4, params.kernel * params.kernel * outGroups)

    // Bias texture: (outGroups, 1)
    const biasTex = this.makeTexture(biasData, outGroups, 1)

    // Output tensor texture: (outW * outGroups, outH)
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
      u_pad_top:      padTop,
      u_pad_left:     padLeft,
      u_activation:   params.activation === 'relu6' ? 1 : params.activation === 'relu' ? 2 : params.activation === 'leaky' ? 3 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outH]
  }
}
