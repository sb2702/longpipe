import type { Backend, Tensor, Op } from '~/model/backend'

// 2D Gaussian blur via two separable 1D passes (horizontal then vertical).
// Backend-agnostic — both backends already implement GaussianBlur1D.
//
// `output` is the result of the vertical pass; both passes' tensors are
// allocated up-front and reused on every run().
export class BlurGaussian {
  private readonly hPass: Op
  private readonly vPass: Op

  constructor(backend: Backend, input: Tensor, sigma: number) {
    this.hPass = backend.ops.GaussianBlur1D(input,            { direction: 'horizontal', sigma })
    this.vPass = backend.ops.GaussianBlur1D(this.hPass.output, { direction: 'vertical',   sigma })
  }

  get output(): Tensor { return this.vPass.output }

  run(): void {
    this.hPass.run()
    this.vPass.run()
  }
}
