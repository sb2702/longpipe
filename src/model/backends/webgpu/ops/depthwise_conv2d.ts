import type { Tensor, DepthwiseParams } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, convOutSize, samePadHalf, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class DepthwiseConv2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly output: WebGPUTensor
  protected pipeline: GPUComputePipeline
  protected bindGroup: GPUBindGroup
  protected dispatchX: number
  protected dispatchY: number
  protected dispatchZ: number

  constructor(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    params: DepthwiseParams,
  ) {
    super(device)
    const outH          = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW          = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const channelGroups = input.c / 4
    const padTop        = params.padding === 'same' ? samePadHalf(input.h, outH, params.kernel, params.stride) : 0
    const padLeft       = params.padding === 'same' ? samePadHalf(input.w, outW, params.kernel, params.stride) : 0

    this.output = makeOutputTensor(device, outH, outW, input.c)

    const uniformBuf = makeUniform(device, [
      input.h, input.w, outH, outW,
      channelGroups,
      params.kernel, params.kernel,
      params.stride, padTop, padLeft,
      params.activation === 'relu6' ? 1 : 0,
      0, // _pad0
    ])

    this.pipeline = pipeline
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cast(input).buffer } },
        { binding: 1, resource: { buffer: cast(weights).buffer } },
        { binding: 2, resource: { buffer: cast(bias).buffer } },
        { binding: 3, resource: { buffer: this.output.buffer } },
        { binding: 4, resource: { buffer: uniformBuf } },
      ],
    })

    this.dispatchX = Math.ceil(outW / 8)
    this.dispatchY = Math.ceil(outH / 8)
    this.dispatchZ = channelGroups
    this.inputs = [input]
  }
}
