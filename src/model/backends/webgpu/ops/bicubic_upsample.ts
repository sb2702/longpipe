import type { Tensor, MLBuffer, UpsampleParams } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGPUTensor, WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op'
import bicubicF32Src from '~/model/backends/webgpu/shaders/bicubic_upsample.wgsl'
import bicubicF16Src from '~/model/backends/webgpu/shaders/bicubic_upsample_f16.wgsl'

export class BicubicUpsampleWebGPU extends WebGPUOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGPUTensor
  protected dispatch: [number, number, number]
  shader: string

  constructor(backend: WebGPUBackend, input: Tensor, params: UpsampleParams) {
    super(backend)
    this.shader = backend.dtype === 'f16' ? bicubicF16Src : bicubicF32Src

    const cGroups = input.c / 4
    this.output = backend.tensor(params.outH, params.outW, input.c)
    this.inputs = [input]

    this.createUniform('params', 'Params')
    this.setUniform('params', new Uint32Array([
      input.h, input.w, params.outH, params.outW, cGroups, 0, 0, 0,
    ]))

    this.defaultSetup()

    this.dispatch = [
      Math.ceil(params.outW / 8),
      Math.ceil(params.outH / 8),
      cGroups,
    ]
  }
}
