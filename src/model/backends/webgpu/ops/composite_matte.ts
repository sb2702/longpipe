import type { Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import compositeMatteF32Src from '~/model/backends/webgpu/shaders/composite_matte.wgsl'
import compositeMatteF16Src from '~/model/backends/webgpu/shaders/composite_matte_f16.wgsl'

// Renders the raw 1-channel alpha matte as a premultiplied white silhouette to
// a canvas swapchain texture. Alpha only — no image, no background. Standalone —
// not a WebGPUOp.
//
// Caller invariants:
//   - alpha.c === 4
//   - canvas.width === alpha.w, canvas.height === alpha.h
//
// Per-frame contract (handled by Backend.presenters.CompositeMatte wrapper):
//   compositor.setOutput(backend.getCurrentDisplayTexture(target))
//   compositor.run()
export class CompositeMatteWebGPU {
  private readonly pipeline:      GPURenderPipeline
  private readonly bindGroup:     GPUBindGroup
  private readonly uniformBuffer: GPUBuffer
  private          outputView:    GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    alpha: Tensor,
  ) {
    const device = backend.device

    // Params: just the alpha width (u32 = 4 bytes; round up to 16 for uniform
    // buffer alignment).
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(16)
    new Uint32Array(ab, 0, 1)[0] = alpha.w
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const src = backend.dtype === 'f16' ? compositeMatteF16Src : compositeMatteF32Src
    const module = device.createShaderModule({ code: src })

    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: backend.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    })

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: (alpha as WebGPUTensor).buffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
      ],
    })
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('CompositeMatteWebGPU.run() called before setOutput()')

    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view:       this.outputView,
        clearValue: [0, 0, 0, 0],
        loadOp:     'clear',
        storeOp:    'store',
      }],
    })
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(6)
    pass.end()
    this.backend.device.queue.submit([enc.finish()])

    // GPUTexture is invalidated after the next browser paint — force the
    // caller to set it again before the next frame.
    this.outputView = null
  }
}
