import type { Backend, Tensor, Presenter, RenderTarget } from '~/model/backend.ts'

// Composites an RGBA image over TRANSPARENCY using a 1-channel alpha mask,
// rendering to the backend's canvas. The matte becomes the canvas alpha channel
// so the subject is isolated on a transparent background (whatever sits behind
// the canvas shows through). Backend-agnostic — dispatches via the `Backend`
// interface, mirroring CompositorSolid.
//
// Caller invariants (enforced inside the per-backend op):
//   - image and alpha are the same h × w (run the upscaler first if needed)
//   - canvas.width === image.w, canvas.height === image.h (no resampling)
export class CompositorTransparent {
  private readonly presenter: Presenter

  constructor(
    backend: Backend,
    image: Tensor,
    alpha: Tensor,
    target: RenderTarget = 'main',
  ) {
    this.presenter = backend.presenters.CompositeTransparent(image, alpha, target)
  }

  run(): void {
    this.presenter.run()
  }
}
