import type { Backend, Tensor, Op } from '~/model/backend.ts'

// Bilinear upscale (align_corners=False) of a Tensor to an arbitrary
// output resolution. Backend-agnostic — wraps backend.ops.BilinearUpsample,
// which already handles arbitrary scale ratios on both backends.
//
// Used at the boundary between the model (low-res alpha) and the compositor
// (alpha at canvas resolution).
export class BilinearUpscaler {
  readonly op: Op

  constructor(backend: Backend, input: Tensor, outH: number, outW: number) {
    this.op = backend.ops.BilinearUpsample(input, { outH, outW })
  }

  get output(): Tensor { return this.op.output }
  run(): void { this.op.run() }
}
