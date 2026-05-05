import type { Backend, Tensor, Presenter } from '~/model/backend.ts'
import { BlurPyramid } from '~/model/effects/blur_pyramid.ts'

// Background-blur composite: blurs the input image with a pyramid (output
// stops at half input resolution) and composites it as the bg via the
// bilinear-sampling presenter. The presenter's final per-pixel scan
// expands the half-res blurred bg back to canvas resolution for free —
// avoiding the most expensive op a full pyramid would have.
export class CompositorBlur {
  private readonly blur:      BlurPyramid
  private readonly presenter: Presenter

  constructor(backend: Backend, image: Tensor, alpha: Tensor, sigma: number) {
    this.blur      = new BlurPyramid(backend, image, sigma)
    this.presenter = backend.presenters.CompositeImageBilinear(image, alpha, this.blur.output)
  }

  run(): void {
    this.blur.run()
    this.presenter.run()
  }
}
