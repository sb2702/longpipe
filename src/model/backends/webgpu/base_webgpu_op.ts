import type { Tensor, Op } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'

export interface WebGPUTensor extends Tensor {
  readonly buffer: GPUBuffer
}

interface UniformDef {
  name: string
  type: string
}

// Base for compute ops. A parallel WebGPURenderOp will handle fragment/render passes.
export abstract class WebGPUOp implements Op {
  abstract readonly inputs: Tensor[]
  abstract readonly output: WebGPUTensor
  protected abstract dispatch: [number, number, number]
  shader: string  = "";
  protected pipeline!: GPUComputePipeline
  protected bindGroup!: GPUBindGroup

  private readonly uniformDefs: UniformDef[] = []
  private readonly uniformBuffers: Record<string, GPUBuffer> = {}

  constructor(protected readonly backend: WebGPUBackend) {}

  protected createUniform(name: string, type: string): void {
    this.uniformDefs.push({ name, type })
  }

  protected setUniform(name: string, data: Float32Array | Uint32Array): void {
    const buf = this.backend.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    })
    if (data instanceof Uint32Array) new Uint32Array(buf.getMappedRange()).set(data)
    else new Float32Array(buf.getMappedRange()).set(data)
    buf.unmap()
    this.uniformBuffers[name] = buf
  }

  // Builds a bind group matching the layout from createStandardShader.
  // Also works with hand-written WGSL as long as bindings follow the same order.
  // Must be called after this.output and this.pipeline are set.
  protected defaultBindGroup(): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = []
    let b = 0
    for (const input of this.inputs) {
      entries.push({ binding: b++, resource: { buffer: (input as WebGPUTensor).buffer } })
    }
    for (const u of this.uniformDefs) {
      entries.push({ binding: b++, resource: { buffer: this.uniformBuffers[u.name] } })
    }
    entries.push({ binding: b, resource: { buffer: this.output.buffer } })
    return this.backend.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    })
  }

  // Creates pipeline from a shader module then builds the bind group.
  protected defaultSetup(): void {

    const shader = this.backend.device.createShaderModule({ code: this.shader });

    this.pipeline = this.backend.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shader, entryPoint: 'main' },
    })
    this.bindGroup = this.defaultBindGroup()
  }

  run(): void {
    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(...this.dispatch)
    pass.end()
    this.backend.device.queue.submit([enc.finish()])
  }
}

export function cast(t: Tensor): WebGPUTensor {
  return t as WebGPUTensor
}
