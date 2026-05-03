import type { Tensor, MLBuffer, Conv2dParams } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLMLBuffer, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import conv2dSrc from '~/model/backends/webgl/shaders/conv2d.glsl'
import { convOutSize, resolvePad } from '~/model/backends/webgpu/ops/conv_utils'

export class Conv2DWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[]
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = conv2dSrc

  constructor(
    backend: WebGLBackend,
    input: Tensor,
    weights: MLBuffer,
    bias: MLBuffer,
    params: Conv2dParams,
  ) {
    super(backend)

    const outH      = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW      = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4
    const padTop    = resolvePad(params.padding, input.h, outH, params.kernel, params.stride)
    const padLeft   = resolvePad(params.padding, input.w, outW, params.kernel, params.stride)

    // Weight texture: (inGroups * 4, kernel² * outGroups)
    const wTexW = inGroups * 4
    const wTexH = params.kernel * params.kernel * outGroups
    const weightTex = this.makeTexture((weights as WebGLMLBuffer).data, wTexW, wTexH)

    // Bias texture: (outGroups, 1)
    const biasTex = this.makeTexture((bias as WebGLMLBuffer).data, outGroups, 1)

    // Output tensor texture: (outW * outGroups, outH)
    const outTexW   = outW * outGroups
    const outTexH   = outH
    const outTexture = this.makeTexture(null, outTexW, outTexH)
    this.output = { h: outH, w: outW, c: params.outChannels, texture: outTexture, texW: outTexW, texH: outTexH }

    this.inputs  = [input]
    this.weights = [weights, bias]

    this.samplers = [
      { name: 'u_input',   texture: (input as WebGLTensor).texture },
      { name: 'u_weights', texture: weightTex },
      { name: 'u_bias',    texture: biasTex },
    ]

    this.uniformInts = {
      u_in_w:        input.w,
      u_in_h:        input.h,
      u_in_c_groups: inGroups,
      u_out_c_groups: outGroups,
      u_kernel_h:    params.kernel,
      u_kernel_w:    params.kernel,
      u_stride:      params.stride,
      u_pad_top:     padTop,
      u_pad_left:    padLeft,
      u_activation:  params.activation === 'relu6' ? 1 : 0,
    }

    this.defaultSetup()
    this.dispatch = [outTexW, outTexH]
  }
}
