import type { Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import compositeTransparentF32Src from '~/model/backends/webgpu/shaders/composite_transparent.wgsl'
import compositeTransparentF16Src from '~/model/backends/webgpu/shaders/composite_transparent_f16.wgsl'

// Composites image + alpha over TRANSPARENCY (the matte becomes the canvas
// alpha) and writes the result to a canvas swapchain texture. Like
// CompositeSolidWebGPU but with no background — the subject is isolated so
// whatever sits behind the canvas shows through. Standalone — not a WebGPUOp.
//
// Caller invariants:
//   - image and alpha are same h × w, c === 4
//   - canvas.width === image.w, canvas.height === image.h
//
// Per-frame contract (handled by Backend.presenters.CompositeTransparent wrapper):
//   compositor.setOutput(backend.getCurrentDisplayTexture(target))
//   compositor.run()
export class CompositeTransparentWebGPU {
  private readonly pipeline:      GPURenderPipeline
  private readonly bindGroup:     GPUBindGroup
  private readonly uniformBuffer: GPUBuffer
  private          outputView:    GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    image: Tensor,
    alpha: Tensor,
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w)
      throw new Error(
        `CompositeTransparent: image (${image.h}×${image.w}) and alpha ` +
        `(${alpha.h}×${alpha.w}) must match. Run the upscaler first.`,
      )

    const device = backend.device

    // Params: just the image width (u32 = 4 bytes; round up to 16 for uniform
    // buffer alignment).
    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(16)
    new Uint32Array(ab, 0, 1)[0] = image.w
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const src = backend.dtype === 'f16' ? compositeTransparentF16Src : compositeTransparentF32Src
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
        { binding: 1, resource: { buffer: (alpha as WebGPUTensor).buffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    })
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('CompositeTransparentWebGPU.run() called before setOutput()')

    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view:       this.outputView,
        // Transparent clear: pixels the full-screen quad doesn't touch (none,
        // here) stay clear rather than opaque black.
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
