import type { Tensor, MLBuffer, FaceTopology, FaceTouchupParams } from '~/model/backend.ts'
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
  private readonly slots: number
  private activeSlots: number
  private outputView: GPUTextureView | null = null

  constructor(
    private readonly backend: WebGPUBackend,
    frame: Tensor,
    landmarks: Tensor,
    box: Tensor,
    topo: FaceTopology,
    params: FaceTouchupParams,
  ) {
    // K faces share one atlas: grid×grid tiles, so the blur/combine/copy passes
    // run ONCE regardless of K — only the two mesh draws scale (instanced).
    // K=1 → grid 1 → the whole atlas, byte-identical to the single-face layout.
    const slots = params.slots ?? 1
    if (slots !== 1 && slots !== 4)
      throw new Error(`FaceTouchup: slots must be 1 or 4, got ${slots}`)
    const grid = slots === 4 ? 2 : 1
    if (box.w * box.h < slots)
      throw new Error(`FaceTouchup: slots ${slots} exceeds the ${box.h}×${box.w} box tensor`)
    if (landmarks.c < slots * 956)
      throw new Error(`FaceTouchup: landmarks tensor holds ${landmarks.c / 956} faces < slots ${slots}`)
    this.slots = slots
    this.activeSlots = slots

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
      const f = new Float32Array(ab, 8, 8)
      f[0] = o.sigma ?? 0; f[1] = params.detail; f[2] = params.strength; f[3] = params.thresh
      f[4] = o.dirX ?? 0; f[5] = o.dirY ?? 0
      f[6] = frame.w; f[7] = frame.h
      new Uint32Array(ab, 40, 2).set([slots, grid])
      device.queue.writeBuffer(buf, 0, ab)
      return buf
    }
    const uPlain = mkUni({})
    const uBlurH = mkUni({ sigma: params.amount, dirX: 1 / ATLAS })
    const uBlurV = mkUni({ sigma: params.amount, dirY: 1 / ATLAS })
    const uBilat = mkUni({ sigma: params.amount, dirX: 1 / ATLAS, dirY: 1 / ATLAS })

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
    const pPass   = mkPipe('vs_pass', 'fs_pass', backend.canvasFormat, false)
    const pComp   = mkPipe('vs_comp', 'fs_comp', backend.canvasFormat, true)

    // Style-dependent smoothing: freq-sep = 3 passes (blur H/V + recombine);
    // bilateral = 1 edge-preserving pass. Both land in `smoothed`.
    const smoothing = (params.style ?? 'freq-sep') === 'bilateral'
      ? (() => {
          const pBilat = mkPipe('vs_quad_fb', 'fs_bilateral', 'rgba8unorm', false)
          return [
            { pipeline: pBilat, target: smoothed.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBilat, uBilat, [{ binding: 4, resource: sampler }, tex(5, atlas)]) },
          ]
        })()
      : (() => {
          const pBlur = mkPipe('vs_quad_fb', 'fs_blur', 'rgba8unorm', false)
          const pComb = mkPipe('vs_quad_fb', 'fs_combine', 'rgba8unorm', false)
          return [
            { pipeline: pBlur, target: ping.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBlur, uBlurH, [{ binding: 4, resource: sampler }, tex(5, atlas)]) },
            { pipeline: pBlur, target: low.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBlur, uBlurV, [{ binding: 4, resource: sampler }, tex(5, ping)]) },
            { pipeline: pComb, target: smoothed.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pComb, uPlain, [{ binding: 4, resource: sampler }, tex(5, atlas), tex(6, low)]) },
          ]
        })()

    this.passes = [
      { pipeline: pUnwrap, target: atlas.createView(), verts: topo.count, mesh: true, load: false,
        bind: bind(pUnwrap, uPlain, [buf(0, frameB), buf(1, lmB), buf(2, boxB)]) },
      ...smoothing,
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
      pass.draw(p.verts, p.mesh ? this.activeSlots : 1)
      pass.end()
    }
    this.device.queue.submit([enc.finish()])
    this.outputView = null
  }
}

// Tensor→Tensor stage form of the touch-up — the composable shape for the
// one-compositor architecture: the retouched frame lands in an output TENSOR
// that the (single, terminal) background compositor consumes as its foreground
// image. Same five passes as the presenter, but the composite renders into a
// float texture (rgba32float / rgba16float by dtype) and a compute blit
// (cs_unpack) copies it into the output buffer — copyTextureToBuffer's
// 256-byte bytesPerRow constraint doesn't hold for arbitrary widths.
export class FaceTouchupStageWebGPU {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  private readonly device: GPUDevice
  private readonly passes: Array<{
    pipeline: GPURenderPipeline
    bind: GPUBindGroup
    target: GPUTextureView
    verts: number
    mesh: boolean
    load: boolean
  }>
  private readonly unpackPipeline: GPUComputePipeline
  private readonly unpackBind: GPUBindGroup
  private readonly unpackDispatch: [number, number]
  private readonly uvBuf: GPUBuffer
  private readonly idxBuf: GPUBuffer
  private readonly meshCount: number
  private readonly slots: number
  private activeSlots: number

  constructor(
    backend: WebGPUBackend,
    frame: Tensor,
    landmarks: Tensor,
    box: Tensor,
    topo: FaceTopology,
    params: FaceTouchupParams,
  ) {
    const ATLAS = 512
    // K faces share one atlas: grid×grid tiles, so the blur/combine/copy passes
    // run ONCE regardless of K — only the two mesh draws scale (instanced).
    // K=1 → grid 1 → the whole atlas, byte-identical to the single-face layout.
    const slots = params.slots ?? 1
    if (slots !== 1 && slots !== 4)
      throw new Error(`FaceTouchup: slots must be 1 or 4, got ${slots}`)
    const grid = slots === 4 ? 2 : 1
    if (box.w * box.h < slots)
      throw new Error(`FaceTouchup: slots ${slots} exceeds the ${box.h}×${box.w} box tensor`)
    if (landmarks.c < slots * 956)
      throw new Error(`FaceTouchup: landmarks tensor holds ${landmarks.c / 956} faces < slots ${slots}`)
    this.slots = slots
    this.activeSlots = slots

    const device = backend.device
    this.device = device
    this.inputs = [frame, landmarks, box]
    this.output = backend.tensor(frame.h, frame.w, 4)
    this.meshCount = topo.count

    this.uvBuf = device.createBuffer({ size: topo.uv.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.uvBuf, 0, topo.uv.buffer as ArrayBuffer, topo.uv.byteOffset, topo.uv.byteLength)
    this.idxBuf = device.createBuffer({ size: topo.idx.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(this.idxBuf, 0, topo.idx.buffer as ArrayBuffer, topo.idx.byteOffset, topo.idx.byteLength)

    const mkAtlasTex = () => device.createTexture({
      size: [ATLAS, ATLAS], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const atlas = mkAtlasTex(), ping = mkAtlasTex(), low = mkAtlasTex(), smoothed = mkAtlasTex()
    // The stage's composite target: float-renderable so the unpack keeps full
    // precision into the f32/f16 tensor buffer.
    const outFormat: GPUTextureFormat = backend.dtype === 'f16' ? 'rgba16float' : 'rgba32float'
    const outTex = device.createTexture({
      size: [frame.w, frame.h], format: outFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    const weight = device.createTexture({
      size: [topo.weightMask.width, topo.weightMask.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    device.queue.copyExternalImageToTexture(
      { source: topo.weightMask }, { texture: weight },
      [topo.weightMask.width, topo.weightMask.height])
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })

    const mkUni = (o: { sigma?: number; dirX?: number; dirY?: number }) => {
      const buf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      const ab = new ArrayBuffer(48)
      const u = new Uint32Array(ab, 0, 2)
      u[0] = frame.w; u[1] = frame.h
      const f = new Float32Array(ab, 8, 8)
      f[0] = o.sigma ?? 0; f[1] = params.detail; f[2] = params.strength; f[3] = params.thresh
      f[4] = o.dirX ?? 0; f[5] = o.dirY ?? 0
      f[6] = frame.w; f[7] = frame.h
      new Uint32Array(ab, 40, 2).set([slots, grid])
      device.queue.writeBuffer(buf, 0, ab)
      return buf
    }
    const uPlain = mkUni({})
    const uBlurH = mkUni({ sigma: params.amount, dirX: 1 / ATLAS })
    const uBlurV = mkUni({ sigma: params.amount, dirY: 1 / ATLAS })
    const uBilat = mkUni({ sigma: params.amount, dirX: 1 / ATLAS, dirY: 1 / ATLAS })

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
    const bind = (pipe: GPURenderPipeline | GPUComputePipeline, uni: GPUBuffer, extra: GPUBindGroupEntry[]) =>
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
    const pPass   = mkPipe('vs_pass', 'fs_pass', outFormat, false)
    const pComp   = mkPipe('vs_comp', 'fs_comp', outFormat, true)

    const smoothing = (params.style ?? 'freq-sep') === 'bilateral'
      ? (() => {
          const pBilat = mkPipe('vs_quad_fb', 'fs_bilateral', 'rgba8unorm', false)
          return [
            { pipeline: pBilat, target: smoothed.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBilat, uBilat, [{ binding: 4, resource: sampler }, tex(5, atlas)]) },
          ]
        })()
      : (() => {
          const pBlur = mkPipe('vs_quad_fb', 'fs_blur', 'rgba8unorm', false)
          const pComb = mkPipe('vs_quad_fb', 'fs_combine', 'rgba8unorm', false)
          return [
            { pipeline: pBlur, target: ping.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBlur, uBlurH, [{ binding: 4, resource: sampler }, tex(5, atlas)]) },
            { pipeline: pBlur, target: low.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pBlur, uBlurV, [{ binding: 4, resource: sampler }, tex(5, ping)]) },
            { pipeline: pComb, target: smoothed.createView(), verts: 3, mesh: false, load: false,
              bind: bind(pComb, uPlain, [{ binding: 4, resource: sampler }, tex(5, atlas), tex(6, low)]) },
          ]
        })()

    this.passes = [
      { pipeline: pUnwrap, target: atlas.createView(), verts: topo.count, mesh: true, load: false,
        bind: bind(pUnwrap, uPlain, [buf(0, frameB), buf(1, lmB), buf(2, boxB)]) },
      ...smoothing,
      { pipeline: pPass, target: outTex.createView(), verts: 3, mesh: false, load: false,
        bind: bind(pPass, uPlain, [buf(0, frameB)]) },
      { pipeline: pComp, target: outTex.createView(), verts: topo.count, mesh: true, load: true,
        bind: bind(pComp, uPlain, [buf(0, frameB), buf(1, lmB), buf(2, boxB),
                                   { binding: 4, resource: sampler }, tex(5, smoothed), tex(6, weight)]) },
    ]

    this.unpackPipeline = device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'cs_unpack' },
    })
    this.unpackBind = bind(this.unpackPipeline, uPlain,
      [tex(5, outTex), buf(7, this.output.buffer)])
    this.unpackDispatch = [Math.ceil(frame.w / 8), Math.ceil(frame.h / 8)]
  }

  // Faces to draw this frame — kept in lockstep with the caller's landmark runs
  // (a live box + stale landmarks smears the old mesh onto the new face).
  setActiveSlots(n: number): void {
    this.activeSlots = Math.max(0, Math.min(n, this.slots))
  }

  run(): void {
    const enc = this.device.createCommandEncoder()
    for (const p of this.passes) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: p.target, clearValue: [0, 0, 0, 1],
          loadOp: p.load ? 'load' : 'clear', storeOp: 'store',
        }],
      })
      pass.setPipeline(p.pipeline)
      pass.setBindGroup(0, p.bind)
      if (p.mesh) {
        pass.setVertexBuffer(0, this.uvBuf)
        pass.setVertexBuffer(1, this.idxBuf)
      }
      pass.draw(p.verts, p.mesh ? this.activeSlots : 1)
      pass.end()
    }
    const c = enc.beginComputePass()
    c.setPipeline(this.unpackPipeline)
    c.setBindGroup(0, this.unpackBind)
    c.dispatchWorkgroups(this.unpackDispatch[0], this.unpackDispatch[1], 1)
    c.end()
    this.device.queue.submit([enc.finish()])
  }
}
