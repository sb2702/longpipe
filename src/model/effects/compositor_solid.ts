import type { Backend, Tensor, Presenter } from '~/model/backend.ts'

// Composites an RGBA image over a solid background color using a 1-channel
// alpha mask, rendering to the backend's canvas. Backend-agnostic — sits
// alongside `~/model/networks` and `~/model/blocks` and dispatches via the
// `Backend` interface.
//
// Caller invariants (enforced inside the per-backend op):
//   - image and alpha are the same h × w (run the upscaler first if needed)
//   - canvas.width === image.w, canvas.height === image.h (no resampling)
export class CompositorSolid {
  private readonly presenter: Presenter

  constructor(
    backend: Backend,
    image: Tensor,
    alpha: Tensor,
    bgColor: [number, number, number],
  ) {
    this.presenter = backend.presenters.CompositeSolid(image, alpha, bgColor)
  }

  run(): void {
    this.presenter.run()
  }
}
