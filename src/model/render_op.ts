import type { Backend, Tensor } from '~/model/backend'
import { BilinearUpscaler } from '~/model/effects/upscale_bilinear'
import { BicubicUpscaler  } from '~/model/effects/upscale_bicubic'
import { CompositorSolid } from '~/model/effects/compositor_solid'
import { CompositorImage } from '~/model/effects/compositor_image'
import { CompositorBlur }  from '~/model/effects/compositor_blur'

export type UpscalerMode = 'bilinear' | 'bicubic'

export type BackgroundConfig =
  | { mode: 'solid'; color: [number, number, number] }
  | { mode: 'image'; image: Tensor }
  | { mode: 'blur';  sigma: number }

export interface NetworkLike {
  readonly output: Tensor
  run(): void
}

export interface RenderOptions {
  upscaler:   UpscalerMode
  background: BackgroundConfig
}

interface UpscalerHandle { readonly output: Tensor; run(): void }
interface CompositorHandle { run(): void }

// Orchestrates a single render pass: network → alpha upscale → composite to
// canvas. The `network` is owned by the caller; `image` is the canvas-
// resolution Tensor used as the foreground in compositing. Effects (upscaler
// mode, background config) can be swapped at runtime without rebuilding the
// network — internal pieces are recreated cheaply (shaders are driver-cached;
// only descriptors/bindings/intermediate tensors are re-allocated).
//
// The contract that the canvas, image, and upscaled alpha must share h × w
// is enforced inside the per-backend compositor ops.
export class RenderOp<N extends NetworkLike = NetworkLike> {
  private upscalerMode: UpscalerMode
  private bgConfig:     BackgroundConfig
  private upscaler:     UpscalerHandle
  private compositor:   CompositorHandle

  constructor(
    private readonly backend: Backend,
    readonly network: N,
    private readonly image: Tensor,
    options: RenderOptions,
  ) {
    this.upscalerMode = options.upscaler
    this.bgConfig     = options.background
    this.upscaler     = this.makeUpscaler()
    this.compositor   = this.makeCompositor()
  }

  setUpscaler(mode: UpscalerMode): void {
    if (mode === this.upscalerMode) return
    this.upscalerMode = mode
    this.upscaler   = this.makeUpscaler()
    this.compositor = this.makeCompositor()  // depends on upscaler.output
  }

  setBackground(config: BackgroundConfig): void {
    this.bgConfig   = config
    this.compositor = this.makeCompositor()
  }

  run(): void {
    this.network.run()
    this.upscaler.run()
    this.compositor.run()
  }

  private makeUpscaler(): UpscalerHandle {
    const { backend, network, image } = this
    return this.upscalerMode === 'bicubic'
      ? new BicubicUpscaler(backend,  network.output, image.h, image.w)
      : new BilinearUpscaler(backend, network.output, image.h, image.w)
  }

  private makeCompositor(): CompositorHandle {
    const { backend, image, upscaler, bgConfig } = this
    switch (bgConfig.mode) {
      case 'solid': return new CompositorSolid(backend, image, upscaler.output, bgConfig.color)
      case 'image': return new CompositorImage(backend, image, upscaler.output, bgConfig.image)
      case 'blur':  return new CompositorBlur(backend,  image, upscaler.output, bgConfig.sigma)
    }
  }
}
