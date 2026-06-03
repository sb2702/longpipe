import type { Backend, Tensor, Presenter } from '~/model/backend.ts'
import { BlurPyramid } from '~/model/effects/blur_pyramid.ts'

// Background-blur composite: blurs the input image with a pyramid (output
// stops at half input resolution) and composites it as the bg via the
// bilinear-sampling presenter. The presenter's final per-pixel scan
// expands the half-res blurred bg back to canvas resolution for free —
// avoiding the most expensive op a full pyramid would have.
// Below this sigma there's effectively no blur to do — and the pyramid floors
// at 2 levels (a visible half-res blur) so it can't represent "off" anyway.
// At/under this, composite the subject over the UNMODIFIED background (strength
// 0 = no blur).
const BLUR_EPS = 0.05

export class CompositorBlur {
  private readonly blur:      BlurPyramid | null
  private readonly presenter: Presenter

  constructor(backend: Backend, image: Tensor, alpha: Tensor, sigma: number) {
    if (sigma <= BLUR_EPS) {
      this.blur      = null
      this.presenter = backend.presenters.CompositeImage(image, alpha, image)
    } else {
      this.blur      = new BlurPyramid(backend, image, sigma)
      this.presenter = backend.presenters.CompositeImageBilinear(image, alpha, this.blur.output)
    }
  }

  run(): void {
    this.blur?.run()
    this.presenter.run()
  }
}
