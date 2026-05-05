import type { Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import compositeImageF32Src from '~/model/backends/webgpu/shaders/composite_image.wgsl'
import compositeImageF16Src from '~/model/backends/webgpu/shaders/composite_image_f16.wgsl'

// Like CompositeSolidWebGPU but bg is a Tensor (e.g. a virtual background or
// a blurred copy of the input). Standalone — produces no Tensor output.
export class CompositeImageWebGPU {
  private readonly pipeline:      GPURenderPipeline
  private readonly bindGroup:     GPUBindGroup
  private readonly uniformBuffer: GPUBuffer
  private          outputView:    GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    image: Tensor,
    alpha: Tensor,
    bg: Tensor,
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w
     || image.h !== bg.h    || image.w !== bg.w) {
      throw new Error(
        `CompositeImage: image (${image.h}×${image.w}), alpha (${alpha.h}×${alpha.w}), ` +
        `and bg (${bg.h}×${bg.w}) must all match. Run upscaler / resizer first.`,
      )
    }

    const device = backend.device

    this.uniformBuffer = device.createBuffer({
      size: 16,  // single u32 + 12 bytes padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(16)
    new Uint32Array(ab, 0, 1)[0] = image.w
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const compositeImageSrc = backend.dtype === 'f16' ? compositeImageF16Src : compositeImageF32Src
    const module = device.createShaderModule({ code: compositeImageSrc })

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
        { binding: 3, resource: { buffer: (bg    as WebGPUTensor).buffer } },
      ],
    })
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('CompositeImageWebGPU.run() called before setOutput()')

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

    this.outputView = null
  }
}
