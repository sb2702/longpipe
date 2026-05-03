import type { Tensor, Conv2dParams } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, convOutSize, samePadHalf, activationCode, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class Conv2DWebGPU extends WebGPUOp {
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
    params: Conv2dParams,
  ) {
    super(device)
    const outH      = convOutSize(input.h, params.kernel, params.stride, params.padding)
    const outW      = convOutSize(input.w, params.kernel, params.stride, params.padding)
    const inGroups  = input.c / 4
    const outGroups = params.outChannels / 4
    const padTop    = params.padding === 'same' ? samePadHalf(input.h, outH, params.kernel, params.stride) : 0
    const padLeft   = params.padding === 'same' ? samePadHalf(input.w, outW, params.kernel, params.stride) : 0

    this.output = makeOutputTensor(device, outH, outW, params.outChannels)

    const uniformBuf = makeUniform(device, [
      input.h, input.w, outH, outW,
      inGroups, outGroups,
      params.kernel, params.kernel,
      params.stride, padTop, padLeft,
      activationCode(params.activation),
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
    this.dispatchZ = outGroups
    this.inputs = [input]
  }
}
