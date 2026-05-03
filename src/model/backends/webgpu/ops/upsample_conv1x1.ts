import type { Tensor, Activation } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, activationCode, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class UpsampleConv1x1WebGPU extends WebGPUOp {
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
    activation: Activation,
    outH: number,
    outW: number,
    outChannels: number,
  ) {
    super(device)
    const inGroups  = input.c / 4
    const outGroups = outChannels / 4

    this.output = makeOutputTensor(device, outH, outW, outChannels)

    const uniformBuf = makeUniform(device, [
      input.h, input.w, outH, outW,
      inGroups, outGroups,
      activationCode(activation), 0,
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
