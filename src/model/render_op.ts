import type { Backend, Tensor, InputOp, ImageSource } from '~/model/backend'
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

// Orchestrates a single render pass: source ingest → network → alpha upscale
// → composite to canvas. Owns a display-resolution Input op (sized to the
// backend's canvas) for the foreground tensor; the caller-supplied
// `networkInput` feeds the model. Both InputOps receive the same source via
// setSource() so a single VideoFrame per tick drives the whole pipeline.
//
// Effects (upscaler mode, background config) can be swapped at runtime
// without rebuilding the network — internal pieces are recreated cheaply
// (shaders are driver-cached; only descriptors/bindings/intermediate tensors
// are re-allocated).
export class RenderOp<N extends NetworkLike = NetworkLike> {
  private upscalerMode: UpscalerMode
  private bgConfig:     BackgroundConfig
  private upscaler:     UpscalerHandle
  private compositor:   CompositorHandle
  private readonly displayInput: InputOp
  private readonly image:        Tensor

  constructor(
    private readonly backend: Backend,
    readonly network: N,
    private readonly networkInput: InputOp,
    options: RenderOptions,
  ) {
    this.upscalerMode = options.upscaler
    this.bgConfig     = options.background

    const canvas = backend.canvas
    this.displayInput = backend.ops.Input(canvas.height, canvas.width)
    this.image        = this.displayInput.output

    this.upscaler   = this.makeUpscaler()
    this.compositor = this.makeCompositor()
  }

  // Stage a single source for the next run(). Both InputOps see the same
  // frame — the network downsamples to its native input resolution, the
  // display path resamples to canvas resolution.
  setSource(src: ImageSource): void {
    this.networkInput.setSource(src)
    this.displayInput.setSource(src)
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
    this.runModel()
    this.runDisplay()
  }

  // Cheap, runs every frame: refresh display tensor at canvas resolution,
  // composite with whatever's currently in the alpha tensor (which persists
  // between model runs).
  runDisplay(): void {
    this.displayInput.run()
    this.compositor.run()
  }

  // Expensive, runs at preset.modelFps: refresh the alpha tensor.
  runModel(): void {
    this.networkInput.run()
    this.network.run()
    this.upscaler.run()
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
