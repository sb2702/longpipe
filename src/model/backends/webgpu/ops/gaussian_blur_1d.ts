import type { Tensor, MLBuffer } from '~/model/backend'
import type { GaussianBlur1DParams } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op'
import blurSrc from '~/model/backends/webgpu/shaders/gaussian_blur_1d.wgsl'

export class GaussianBlur1DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader = blurSrc

  constructor(backend: WebGPUBackend, input: Tensor, params: GaussianBlur1DParams) {
    super(backend)

    const cGroups = input.c / 4
    this.output = backend.tensor(input.h, input.w, input.c)
    this.inputs = [input]

    const stepX = params.direction === 'horizontal' ? 1 : 0
    const stepY = params.direction === 'vertical'   ? 1 : 0

    // Params struct: 32 bytes — 5 × 4 (u32/i32) + 1 × 4 (f32) + 8 padding.
    const ab = new ArrayBuffer(32)
    new Uint32Array(ab,  0, 1)[0] = input.w
    new Uint32Array(ab,  4, 1)[0] = input.h
    new Uint32Array(ab,  8, 1)[0] = cGroups
    new Int32Array(ab,  12, 1)[0] = stepX
    new Int32Array(ab,  16, 1)[0] = stepY
    new Float32Array(ab, 20, 1)[0] = params.sigma

    this.createUniform('params', 'Params')
    // Pre-built ArrayBuffer with mixed types — write directly through
    // setUniform's Float32Array overload (the contents are reinterpreted bytes).
    this.setUniform('params', new Float32Array(ab))

    this.defaultSetup()

    this.dispatch = [
      Math.ceil(input.w / 8),
      Math.ceil(input.h / 8),
      cGroups,
    ]
  }
}
