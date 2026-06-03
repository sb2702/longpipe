import type { Backend, Tensor, Presenter, RenderTarget } from '~/model/backend.ts'

// Composites an RGBA image over a background image (also as a Tensor) using a
// 1-channel alpha mask, rendering to the backend's canvas. Used for virtual
// background replacement (caller provides the bg image as a Tensor) and as
// the inner step of CompositorBlur (where bg is a blurred copy of the input).
//
// Caller invariants (enforced inside the per-backend op):
//   - image, alpha, and bg all share the same h × w
//   - canvas.width === image.w, canvas.height === image.h
export class CompositorImage {
  private readonly presenter: Presenter

  constructor(backend: Backend, image: Tensor, alpha: Tensor, bg: Tensor, target: RenderTarget = 'main') {
    this.presenter = backend.presenters.CompositeImage(image, alpha, bg, target)
  }

  run(): void {
    this.presenter.run()
  }
}
