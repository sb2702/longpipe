import type { Tensor } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import compositeSolidF32Src from '~/model/backends/webgpu/shaders/composite_solid.wgsl'
import compositeSolidF16Src from '~/model/backends/webgpu/shaders/composite_solid_f16.wgsl'

// Composites image + alpha over a solid background and writes the result to a
// canvas swapchain texture. Standalone — not a WebGPUOp (those are compute-
// only and produce a Tensor; this produces nothing, it presents).
//
// Caller invariants:
//   - image and alpha are same h × w, c === 4
//   - canvas.width === image.w, canvas.height === image.h
//
// Per-frame contract:
//   compositor.setOutput(backend.getCurrentDisplayTexture())
//   compositor.run()
export class CompositeSolidWebGPU {
  private readonly pipeline:      GPURenderPipeline
  private readonly bindGroup:     GPUBindGroup
  private readonly uniformBuffer: GPUBuffer
  private          outputView:    GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    image: Tensor,
    alpha: Tensor,
    bgColor: [number, number, number],
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w)
      throw new Error(
        `CompositeSolid: image (${image.h}×${image.w}) and alpha ` +
        `(${alpha.h}×${alpha.w}) must match. Run the upscaler first.`,
      )

    const device = backend.device

    // Params: 32-byte struct, std140-style.
    //   [ 0..4)   width   (u32)
    //   [4..16)   padding
    //   [16..32)  bgColor (vec4<f32>; .a unused)
    this.uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const ab = new ArrayBuffer(32)
    new Uint32Array(ab,  0, 1)[0] = image.w
    new Float32Array(ab, 16, 4).set([bgColor[0], bgColor[1], bgColor[2], 0])
    device.queue.writeBuffer(this.uniformBuffer, 0, ab)

    const compositeSolidSrc = backend.dtype === 'f16' ? compositeSolidF16Src : compositeSolidF32Src
    const module = device.createShaderModule({ code: compositeSolidSrc })

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
      throw new Error('CompositeSolidWebGPU.run() called before setOutput()')

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
