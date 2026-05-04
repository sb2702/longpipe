import type { Tensor, MLBuffer, Conv2dParams } from '~/model/backend'
import type { Conv2DWeights } from '~/model/weights'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import conv2dAddSrc from '~/model/backends/webgl/shaders/conv2d_add.glsl'
import { convOutSize, resolvePad } from '~/model/backends/webgpu/ops/conv_utils'
import { toUploadView } from '~/utils/weights'

export class Conv2dAddWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = conv2dAddSrc

  constructor(backend: WebGLBackend, input: Tensor, skip: Tensor, w: Conv2DWeights, params: Conv2dParams) {
    super(backend)

    const outH      = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW      = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4
    const padTop    = resolvePad(params.padding, input.h, outH, params.kernel, params.stride)
    const padLeft   = resolvePad(params.padding, input.w, outW, params.kernel, params.stride)

    const weightData = toUploadView(w.weights)
    const biasData   = toUploadView(w.bias)

    const weightTex = this.makeTexture(weightData, inGroups * 4, params.kernel * params.kernel * outGroups)
    const biasTex   = this.makeTexture(biasData, outGroups, 1)

    const outTexW    = outW * outGroups
    const outTexture = this.makeTexture(null, outTexW, outH)
    this.output = { h: outH, w: outW, c: params.outChannels, texture: outTexture, texW: outTexW, texH: outH }
    this.inputs  = [input, skip]
    this.weights = []

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_skip',    texture: (skip  as WebGLTensor).texture },
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
      u_activation:   params.activation === 'relu6' ? 1 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outH]
  }
}
