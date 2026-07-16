import type { Tensor, LandmarkOverlayParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/landmark_overlay.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/landmark_overlay_f16.wgsl'

// Landmark overlay presenter — image pass + vertex-pulled landmark dots. The
// vertex shader reads the landmark tensor (LandmarkNet output) and box tensor
// directly from GPU buffers: no readback anywhere in the path. Standalone like
// the compositors (produces no Tensor; presents to a canvas swapchain texture).
//
// Caller invariants: canvas === image w × h; landmarks is the 1×1×(count·2)
// LandmarkNet output; box is the 1×1×4 FaceBoxFromHeatmaps output.
export class LandmarkOverlayWebGPU {
  private readonly imgPipeline: GPURenderPipeline
  private readonly ptsPipeline: GPURenderPipeline
  private readonly imgBindGroup: GPUBindGroup
  private readonly ptsBindGroup: GPUBindGroup
  private readonly count: number
  private readonly drawImage: boolean
  private outputView: GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    image: Tensor,
    landmarks: Tensor,
    box: Tensor,
    params: LandmarkOverlayParams,
  ) {
    if (landmarks.c < params.count * 2)
      throw new Error(`LandmarkOverlay: landmarks tensor holds ${landmarks.c / 2} points < count ${params.count}`)
    const slot = params.slot ?? 0
    if (slot >= box.w * box.h)
      throw new Error(`LandmarkOverlay: slot ${slot} out of range for a ${box.h}×${box.w} box tensor`)
    this.count = params.count
    this.drawImage = params.drawImage ?? true
    const device = backend.device

    // Params: { img_w, count: u32, thresh, point_size: f32, slot: u32,
    //           color: vec4, canvas: vec4 } — vec4s are 16B-aligned, so slot's
    // block pads out to 32 and the struct is 64B.
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    const ab = new ArrayBuffer(64)
    const u = new Uint32Array(ab, 0, 2)
    u[0] = image.w; u[1] = params.count
    const f = new Float32Array(ab, 8, 2)
    f[0] = params.thresh; f[1] = params.pointSize
    new Uint32Array(ab, 16, 1)[0] = slot
    new Float32Array(ab, 32, 4).set([...params.color, 1])
    new Float32Array(ab, 48, 4).set([image.w, image.h, 0, 0])
    device.queue.writeBuffer(uniform, 0, ab)

    const module = device.createShaderModule({ code: backend.dtype === 'f16' ? f16Src : f32Src })
    const target = [{ format: backend.canvasFormat }]
    this.imgPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_img' },
      fragment: { module, entryPoint: 'fs_img', targets: target },
      primitive: { topology: 'triangle-list' },
    })
    this.ptsPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: 'vs_pts' },
      fragment: { module, entryPoint: 'fs_pts', targets: target },
      primitive: { topology: 'triangle-list' },
    })

    // 'auto' layouts only include the bindings each entry-point pair actually
    // uses — hence two bind groups over the same resources.
    this.imgBindGroup = device.createBindGroup({
      layout: this.imgPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: (image as WebGPUTensor).buffer } },
        { binding: 3, resource: { buffer: uniform } },
      ],
    })
    this.ptsBindGroup = device.createBindGroup({
      layout: this.ptsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: (landmarks as WebGPUTensor).buffer } },
        { binding: 2, resource: { buffer: (box as WebGPUTensor).buffer } },
        { binding: 3, resource: { buffer: uniform } },
      ],
    })
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('LandmarkOverlayWebGPU.run() called before setOutput()')

    // drawImage=false layers this face's dots onto whatever a previous overlay
    // left in the target (multi-face) — load, don't clear, and skip the image.
    const enc = this.backend.device.createCommandEncoder()
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.outputView,
        clearValue: [0, 0, 0, 1],
        loadOp: this.drawImage ? 'clear' : 'load',
        storeOp: 'store',
      }],
    })
    if (this.drawImage) {
      pass.setPipeline(this.imgPipeline)
      pass.setBindGroup(0, this.imgBindGroup)
      pass.draw(6)
    }
    pass.setPipeline(this.ptsPipeline)
    pass.setBindGroup(0, this.ptsBindGroup)
    pass.draw(this.count * 6)
    pass.end()
    this.backend.device.queue.submit([enc.finish()])
    this.outputView = null
  }
}
