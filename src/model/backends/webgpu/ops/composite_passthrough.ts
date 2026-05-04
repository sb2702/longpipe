import type { Tensor } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'
import compositePassthroughF32Src from '~/model/backends/webgpu/shaders/composite_passthrough.wgsl'
import compositePassthroughF16Src from '~/model/backends/webgpu/shaders/composite_passthrough_f16.wgsl'

// Passthrough "compositor": writes the image directly to the canvas
// swapchain texture. No alpha, no background. Standalone — not a WebGPUOp.
//
// Used by RenderOp when the renderer is in disabled state (true passthrough
// at the GPU level: input frame in, same frame out on the canvas).
//
// Caller invariants:
//   - image.c === 4
//   - canvas.width === image.w, canvas.height === image.h
//
// Per-frame contract (handled by Backend.presenters.CompositePassthrough wrapper):
//   compositor.setOutput(backend.getCurrentDisplayTexture())
//   compositor.run()
export class CompositePassthroughWebGPU {
  private readonly pipeline:      GPURenderPipeline
  private readonly bindGroup:     GPUBindGroup
  private readonly uniformBuffer: GPUBuffer
  private          outputView:    GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    image: Tensor,
  ) {
    const device = backend.device

    // Params: just the image width (u32 = 4 bytes; round up to 16 for
    // uniform buffer alignment).
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(16)
    new Uint32Array(ab, 0, 1)[0] = image.w
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const src = backend.dtype === 'f16' ? compositePassthroughF16Src : compositePassthroughF32Src
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
        { binding: 0, resource: { buffer: (image as WebGPUTensor).buffer } },
        { binding: 1, resource: { buffer: this.uniformBuffer } },
      ],
    })
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('CompositePassthroughWebGPU.run() called before setOutput()')

    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view:       this.outputView,
        clearValue: [0, 0, 0, 1],
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
