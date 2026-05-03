import type { Tensor } from '~/model/backend'
import { WebGPUTensor, WebGPUOp, makeUniform, makeOutputTensor, cast } from '~/model/backends/webgpu/base'

export class ConcatWebGPU extends WebGPUOp {
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
    const aGroups   = a.c / 4
    const bGroups   = b.c / 4
    const outGroups = aGroups + bGroups
    const outC      = a.c + b.c

    this.output = makeOutputTensor(device, a.h, a.w, outC)

    const uniformBuf = makeUniform(device, [
      a.h, a.w,
      aGroups, bGroups, outGroups,
      0, 0, 0,
    ])

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

    this.dispatchX = Math.ceil(a.w / 8)
    this.dispatchY = Math.ceil(a.h / 8)
    this.dispatchZ = outGroups
    this.inputs = [a, b]
  }
}
