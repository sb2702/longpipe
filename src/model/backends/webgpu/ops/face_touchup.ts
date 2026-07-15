import type { Tensor, FaceTopology, FaceTouchupParams } from '~/model/backend.ts'
import type { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op.ts'
import f32Src from '~/model/backends/webgpu/shaders/face_touchup.wgsl'
import f16Src from '~/model/backends/webgpu/shaders/face_touchup_f16.wgsl'

const ATLAS = 512

// UV-space face touch-up presenter — five passes per frame (unwrap → blur H →
// blur V → freq-sep combine → composite), all GPU-resident: the mesh vertex
// shaders pull landmark positions from the LandmarkNet output buffer and the
// box tensor. Standalone like the compositors (presents to the canvas).
//
// One uniform buffer PER PASS CONFIG, written once at construction —
// queue.writeBuffer is queue-ordered, so a single shared UBO written per pass
// would make every pass read the last value (the ar-scope gotcha).
export class FaceTouchupWebGPU {
  private readonly device: GPUDevice
  private readonly passes: Array<{
    pipeline: GPURenderPipeline
    bind: GPUBindGroup
    target: GPUTextureView | null   // null = canvas (set per frame)
    verts: number
    mesh: boolean
    load: boolean
  }>
  private readonly uvBuf: GPUBuffer
  private readonly idxBuf: GPUBuffer
  private outputView: GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    frame: Tensor,
    landmarks: Tensor,
    box: Tensor,
    topo: FaceTopology,
    params: FaceTouchupParams,
  ) {
    const device = backend.device
    this.device = device

    // Static mesh vertex buffers.
    this.uvBuf = device.createBuffer({ size: topo.uv.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.uvBuf, 0, topo.uv.buffer as ArrayBuffer, topo.uv.byteOffset, topo.uv.byteLength)
    this.idxBuf = device.createBuffer({ size: topo.idx.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.idxBuf, 0, topo.idx.buffer as ArrayBuffer, topo.idx.byteOffset, topo.idx.byteLength)

    // Atlas render targets + weight mask.
    const mkTex = () => device.createTexture({
      size: [ATLAS, ATLAS], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const atlas = mkTex(), ping = mkTex(), low = mkTex(), smoothed = mkTex()
    const weight = device.createTexture({
      size: [topo.weightMask.width, topo.weightMask.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture(
      { source: topo.weightMask }, { texture: weight },
      [topo.weightMask.width, topo.weightMask.height])
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })

    // Per-pass uniforms (shared struct; only the relevant fields vary).
    const mkUni = (o: { sigma?: number; dirX?: number; dirY?: number }) => {
      const buf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      const ab = new ArrayBuffer(48)
      const u = new Uint32Array(ab, 0, 2)
      u[0] = frame.w; u[1] = frame.h
      const f = new Float32Array(ab, 8, 10)
      f[0] = o.sigma ?? 0; f[1] = params.detail; f[2] = params.strength; f[3] = params.thresh
      f[4] = o.dirX ?? 0; f[5] = o.dirY ?? 0
      f[6] = frame.w; f[7] = frame.h
      device.queue.writeBuffer(buf, 0, ab)
      return buf
    }
    const uPlain = mkUni({})
    const uBlurH = mkUni({ sigma: params.amount, dirX: 1 / ATLAS })
    const uBlurV = mkUni({ sigma: params.amount, dirY: 1 / ATLAS })

    const module = device.createShaderModule({ code: backend.dtype === 'f16' ? f16Src : f32Src })
    const meshBuffers: GPUVertexBufferLayout[] = [
      { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
      { arrayStride: 4, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32' }] },
    ]
    const mkPipe = (vs: string, fs: string, format: GPUTextureFormat, mesh: boolean) =>
      device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: vs, buffers: mesh ? meshBuffers : [] },
        fragment: { module, entryPoint: fs, targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      })
    const bind = (pipe: GPURenderPipeline, uni: GPUBuffer, extra: GPUBindGroupEntry[]) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [...extra, { binding: 3, resource: { buffer: uni } }],
      })
    const buf = (binding: number, b: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer: b } })
    const tex = (binding: number, t: GPUTexture): GPUBindGroupEntry => ({ binding, resource: t.createView() })

    const frameB = (frame as WebGPUTensor).buffer
    const lmB = (landmarks as WebGPUTensor).buffer
    const boxB = (box as WebGPUTensor).buffer

    const pUnwrap = mkPipe('vs_unwrap', 'fs_unwrap', 'rgba8unorm', true)
    const pBlur   = mkPipe('vs_quad_fb', 'fs_blur', 'rgba8unorm', false)
    const pComb   = mkPipe('vs_quad_fb', 'fs_combine', 'rgba8unorm', false)
    const pPass   = mkPipe('vs_pass', 'fs_pass', backend.canvasFormat, false)
    const pComp   = mkPipe('vs_comp', 'fs_comp', backend.canvasFormat, true)

    this.passes = [
      { pipeline: pUnwrap, target: atlas.createView(), verts: topo.count, mesh: true, load: false,
        bind: bind(pUnwrap, uPlain, [buf(0, frameB), buf(1, lmB), buf(2, boxB)]) },
      { pipeline: pBlur, target: ping.createView(), verts: 3, mesh: false, load: false,
        bind: bind(pBlur, uBlurH, [{ binding: 4, resource: sampler }, tex(5, atlas)]) },
      { pipeline: pBlur, target: low.createView(), verts: 3, mesh: false, load: false,
        bind: bind(pBlur, uBlurV, [{ binding: 4, resource: sampler }, tex(5, ping)]) },
      { pipeline: pComb, target: smoothed.createView(), verts: 3, mesh: false, load: false,
        bind: bind(pComb, uPlain, [{ binding: 4, resource: sampler }, tex(5, atlas), tex(6, low)]) },
      { pipeline: pPass, target: null, verts: 3, mesh: false, load: false,
        bind: bind(pPass, uPlain, [buf(0, frameB)]) },
      { pipeline: pComp, target: null, verts: topo.count, mesh: true, load: true,
        bind: bind(pComp, uPlain, [buf(0, frameB), buf(1, lmB), buf(2, boxB),
                                   { binding: 4, resource: sampler }, tex(5, smoothed), tex(6, weight)]) },
    ]
  }

  setOutput(texture: GPUTexture): void {
    this.outputView = texture.createView()
  }

  run(): void {
    if (!this.outputView)
      throw new Error('FaceTouchupWebGPU.run() called before setOutput()')

    const enc = this.device.createCommandEncoder()
    for (const p of this.passes) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: p.target ?? this.outputView,
          clearValue: [0, 0, 0, 1],
          loadOp: p.load ? 'load' : 'clear',
          storeOp: 'store',
        }],
      })
      pass.setPipeline(p.pipeline)
      pass.setBindGroup(0, p.bind)
      if (p.mesh) {
        pass.setVertexBuffer(0, this.uvBuf)
        pass.setVertexBuffer(1, this.idxBuf)
      }
      pass.draw(p.verts)
      pass.end()
    }
    this.device.queue.submit([enc.finish()])
    this.outputView = null
  }
}
