import type { Tensor, Op } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'

export class WebGPUTensor implements Tensor {
  constructor(
    readonly h: number,
    readonly w: number,
    readonly c: number,
    readonly buffer: GPUBuffer,
  ) {}
}

export abstract class WebGPUOp implements Op {
  abstract readonly inputs: Tensor[]
  abstract readonly output: WebGPUTensor
  protected abstract pipeline: GPUComputePipeline
  protected abstract bindGroup: GPUBindGroup
  protected abstract dispatchX: number
  protected abstract dispatchY: number
  protected abstract dispatchZ: number

  constructor(protected readonly backend: WebGPUBackend) {}

  run(): void {
    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.dispatchX, this.dispatchY, this.dispatchZ)
    pass.end()
    this.backend.device.queue.submit([enc.finish()])
  }
}

export function cast(t: Tensor): WebGPUTensor {
  return t as WebGPUTensor
}
