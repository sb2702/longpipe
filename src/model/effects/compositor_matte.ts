import type { Backend, Tensor, Presenter, RenderTarget } from '~/model/backend.ts'

// Renders the raw 1-channel alpha matte as a premultiplied white silhouette to
// the backend's canvas — a debug view and a reusable mask (composite it against
// your own full-resolution source). Alpha only — no image, no background.
// Backend-agnostic — dispatches via the `Backend` interface.
//
// Caller invariants (enforced inside the per-backend op):
//   - canvas.width === alpha.w, canvas.height === alpha.h (no resampling)
export class CompositorMatte {
  private readonly presenter: Presenter

  constructor(
    backend: Backend,
    alpha: Tensor,
    target: RenderTarget = 'main',
  ) {
    this.presenter = backend.presenters.CompositeMatte(alpha, target)
  }

  run(): void {
    this.presenter.run()
  }
}
