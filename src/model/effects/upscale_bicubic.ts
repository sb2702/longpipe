import type { Backend, Tensor, Op } from '~/model/backend'

// Bicubic upscale (Keys cubic, a=-0.75, align_corners=False) of a Tensor to
// an arbitrary output resolution. Backend-agnostic — wraps the per-backend
// BicubicUpsample op. Sharper alpha edges than BilinearUpscaler at the cost
// of 16 vs 4 taps per output pixel; worth it for the larger presets where
// the alpha→canvas scale ratio is biggest (e.g. xl, 288→1024).
export class BicubicUpscaler {
  readonly op: Op

  constructor(backend: Backend, input: Tensor, outH: number, outW: number) {
    this.op = backend.ops.BicubicUpsample(input, { outH, outW })
  }

  get output(): Tensor { return this.op.output }
  run(): void { this.op.run() }
}
