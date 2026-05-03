import type { Tensor } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class UpsampleWebGPU extends WebGPUOp {
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
    outH: number,
    outW: number,
  ) {
    super(device)
    const channelGroups = input.c / 4

    this.output = makeOutputTensor(device, outH, outW, input.c)

    const uniformBuf = makeUniform(device, [
      input.h, input.w, outH, outW,
      channelGroups, 0, 0, 0,
    ])

    this.pipeline = pipeline
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cast(input).buffer } },
        { binding: 1, resource: { buffer: this.output.buffer } },
        { binding: 2, resource: { buffer: uniformBuf } },
      ],
    })

    this.dispatchX = Math.ceil(outW / 8)
    this.dispatchY = Math.ceil(outH / 8)
    this.dispatchZ = channelGroups
    this.inputs = [input]
  }
}
