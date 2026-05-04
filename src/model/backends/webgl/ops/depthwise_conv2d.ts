import type { Tensor, MLBuffer, DepthwiseParams } from '~/model/backend'
import type { DepthwiseWeights } from '~/model/weights'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import depthwiseSrc from '~/model/backends/webgl/shaders/depthwise_conv2d.glsl'
import { convOutSize, resolvePad } from '~/model/backends/webgpu/ops/conv_utils'
import { toUploadView } from '~/utils/weights'

export class DepthwiseConv2DWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = depthwiseSrc

  constructor(backend: WebGLBackend, input: Tensor, w: DepthwiseWeights, params: DepthwiseParams) {
    super(backend)

    const outH          = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW          = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const channelGroups = input.c / 4
    const padTop        = resolvePad(params.padding, input.h, outH, params.kernel, params.stride)
    const padLeft       = resolvePad(params.padding, input.w, outW, params.kernel, params.stride)

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    // Weight texture: (channelGroups, kernel²)
    const weightTex = this.makeTexture(weightData, channelGroups, params.kernel * params.kernel)

    // Bias texture: (channelGroups, 1)
    const biasTex = this.makeTexture(biasData, channelGroups, 1)

    // Output tensor texture: (outW * channelGroups, outH)
    const outTexW    = outW * channelGroups
    const outTexture = this.makeTexture(null, outTexW, outH)
    this.output  = { h: outH, w: outW, c: input.c, texture: outTexture, texW: outTexW, texH: outH }
    this.inputs  = [input]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = {
      u_in_w:        input.w,
      u_in_h:        input.h,
      u_c_groups:    channelGroups,
      u_kernel_h:    params.kernel,
      u_kernel_w:    params.kernel,
      u_stride:      params.stride,
      u_pad_top:     padTop,
      u_pad_left:    padLeft,
      u_apply_relu6: params.activation === 'relu6' ? 1 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outH]
  }
}
