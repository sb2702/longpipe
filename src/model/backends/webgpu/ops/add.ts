import type { Tensor } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class AddWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly output: WebGPUTensor
  protected pipeline: GPUComputePipeline
  protected bindGroup: GPUBindGroup
  protected dispatchX: number
  protected dispatchY: number
  protected dispatchZ: number

  constructor(device: GPUDevice, pipeline: GPUComputePipeline, inputs: [Tensor, Tensor]) {
    super(device)
    const [a, b] = inputs
    const size = a.h * a.w * a.c  // total f32 elements

    this.output = makeOutputTensor(device, a.h, a.w, a.c)

    const uniformBuf = makeUniform(device, [size, 0, 0, 0])

    this.pipeline = pipeline
    this.bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cast(a).buffer } },
        { binding: 1, resource: { buffer: cast(b).buffer } },
        { binding: 2, resource: { buffer: this.output.buffer } },
        { binding: 3, resource: { buffer: uniformBuf } },
      ],
    })

    this.dispatchX = Math.ceil(size / 256)
    this.dispatchY = 1
    this.dispatchZ = 1
    this.inputs = [a, b]
  }
}
