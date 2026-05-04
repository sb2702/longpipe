import type { Backend, Tensor } from '~/model/backend'
import { BlurGaussian } from '~/model/effects/blur_gaussian'
import { CompositorImage } from '~/model/effects/compositor_image'

// Background-blur composite: blurs the input image and uses that as the bg
// for a composite_image pass. Caller invariants follow CompositorImage:
// image, alpha, and the (internally produced) blurred image all share h × w.
export class CompositorBlur {
  private readonly blur: BlurGaussian
  private readonly comp: CompositorImage

  constructor(backend: Backend, image: Tensor, alpha: Tensor, sigma: number) {
    this.blur = new BlurGaussian(backend, image, sigma)
    this.comp = new CompositorImage(backend, image, alpha, this.blur.output)
  }

  run(): void {
    this.blur.run()
    this.comp.run()
  }
}
