import type { Tensor } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class SigmoidWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly output: WebGPUTensor
  protected pipeline: GPUComputePipeline
  protected bindGroup: GPUBindGroup
  protected dispatchX: number
  protected dispatchY: number
  protected dispatchZ: number

  constructor(device: GPUDevice, pipeline: GPUComputePipeline, input: Tensor) {
    super(device)
    const nGroups = input.h * input.w * (input.c / 4)  // total vec4 elements

    this.output = makeOutputTensor(device, input.h, input.w, input.c)

    const uniformBuf = makeUniform(device, [nGroups, 0, 0, 0])

    this.pipeline = pipeline
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cast(input).buffer } },
        { binding: 1, resource: { buffer: this.output.buffer } },
        { binding: 2, resource: { buffer: uniformBuf } },
      ],
    })

    this.dispatchX = Math.ceil(nGroups / 256)
    this.dispatchY = 1
    this.dispatchZ = 1
    this.inputs = [input]
  }
}
