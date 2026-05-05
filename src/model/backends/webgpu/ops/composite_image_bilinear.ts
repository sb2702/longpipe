import type { Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import compositeBilinearF32Src from '~/model/backends/webgpu/shaders/composite_image_bilinear.wgsl'
import compositeBilinearF16Src from '~/model/backends/webgpu/shaders/composite_image_bilinear_f16.wgsl'

// Like CompositeImageWebGPU but bg is bilinearly sampled — bg may be smaller
// than (image, alpha). Used by CompositorBlur to drop the final full-res
// upsample of the blur pyramid; this shader's per-pixel scan absorbs that work
// for free.
export class CompositeImageBilinearWebGPU {
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
    if (image.h !== alpha.h || image.w !== alpha.w) {
      throw new Error(
        `CompositeImageBilinear: image (${image.h}×${image.w}) and alpha (${alpha.h}×${alpha.w}) must match.`,
      )
    }

    const device = backend.device

    this.uniformBuffer = device.createBuffer({
      size: 16,  // 4 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(16)
    const u = new Uint32Array(ab)
    u[0] = image.w
    u[1] = image.h
    u[2] = bg.w
    u[3] = bg.h
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const shaderSrc = backend.dtype === 'f16' ? compositeBilinearF16Src : compositeBilinearF32Src
    const module = device.createShaderModule({ code: shaderSrc })

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
      throw new Error('CompositeImageBilinearWebGPU.run() called before setOutput()')

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
