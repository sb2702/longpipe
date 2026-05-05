import type { Backend, Tensor, InputOp, ImageSource } from '~/model/backend.ts'
import { BilinearUpscaler } from '~/model/effects/upscale_bilinear.ts'
import { BicubicUpscaler  } from '~/model/effects/upscale_bicubic.ts'
import { CompositorSolid } from '~/model/effects/compositor_solid.ts'
import { CompositorImage } from '~/model/effects/compositor_image.ts'
import { CompositorBlur }  from '~/model/effects/compositor_blur.ts'

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
// backend's canvas) for the foreground tensor.
//
// Two-phase construction:
//   1. constructor(backend): builds displayInput + passthroughCompositor
//      only — enough to runPassthrough() while the worker is still booting
//      (no preset, no weights, no network needed).
//   2. attachNetwork(network, networkInput, options): wires upscaler +
//      effect compositor. After this, runDisplay() / runModel() / run()
//      are valid. Can be called multiple times (e.g. adaptive preset
//      swaps); each call replaces the network-dependent pieces.
//
// runPassthrough() works at any time. setBackground / setUpscaler are
// safe to call before attachNetwork — they update the stored config and
// take effect on attach.
export class RenderOp<N extends NetworkLike = NetworkLike> {
  // Always built — passthrough path only needs these.
  private readonly displayInput: InputOp
  private readonly image:        Tensor
  private readonly passthroughCompositor: CompositorHandle

  // Network-dependent pieces. Null until attachNetwork().
  private network:      N | null = null
  private networkInput: InputOp | null = null
  private upscaler:     UpscalerHandle | null = null
  private compositor:   CompositorHandle | null = null

  // Stored config — applied lazily when the compositor is (re)built.
  private upscalerMode: UpscalerMode      = 'bilinear'
  private bgConfig:     BackgroundConfig  = { mode: 'solid', color: [0, 0, 0] }

  constructor(private readonly backend: Backend) {
    const canvas = backend.canvas
    this.displayInput = backend.ops.Input(canvas.height, canvas.width)
    this.image        = this.displayInput.output
    this.passthroughCompositor = backend.presenters.CompositePassthrough(this.image)
  }

  // Wire (or rewire) the network chain. Replaces upscaler + compositor;
  // keeps displayInput + passthroughCompositor.
  attachNetwork(network: N, networkInput: InputOp, options: RenderOptions): void {
    this.network      = network
    this.networkInput = networkInput
    this.upscalerMode = options.upscaler
    this.bgConfig     = options.background
    this.upscaler     = this.makeUpscaler()
    this.compositor   = this.makeCompositor()
  }

  hasNetwork(): boolean {
    return this.network !== null
  }

  // Stage a single source for the next run(). Both InputOps see the same
  // frame — the network downsamples to its native input resolution, the
  // display path resamples to canvas resolution. networkInput is optional
  // before attachNetwork so passthrough still works.
  setSource(src: ImageSource): void {
    this.networkInput?.setSource(src)
    this.displayInput.setSource(src)
  }

  setUpscaler(mode: UpscalerMode): void {
    if (mode === this.upscalerMode) return
    this.upscalerMode = mode
    if (this.network) {
      this.upscaler   = this.makeUpscaler()
      this.compositor = this.makeCompositor()  // depends on upscaler.output
    }
  }

  setBackground(config: BackgroundConfig): void {
    this.bgConfig = config
    if (this.network) this.compositor = this.makeCompositor()
  }

  run(): void {
    this.runModel()
    this.runDisplay()
  }

  // Cheap, runs every frame: refresh display tensor at canvas resolution,
  // composite with whatever's currently in the alpha tensor (which persists
  // between model runs).
  runDisplay(): void {
    if (!this.compositor) throw new Error('RenderOp.runDisplay called before attachNetwork')
    this.displayInput.run()
    this.compositor.run()
  }

  // True passthrough: writes the input image directly to the canvas. Used
  // by the renderer when disabled OR while booting (no network attached
  // yet). setSource() is still required (display input needs a fresh
  // frame); model + alpha pipeline is skipped entirely.
  runPassthrough(): void {
    this.displayInput.run()
    this.passthroughCompositor.run()
  }

  // Expensive, runs at preset.modelFps: refresh the alpha tensor.
  runModel(): void {
    if (!this.network || !this.networkInput || !this.upscaler) {
      throw new Error('RenderOp.runModel called before attachNetwork')
    }
    this.networkInput.run()
    this.network.run()
    this.upscaler.run()
  }

  private makeUpscaler(): UpscalerHandle {
    if (!this.network) throw new Error('makeUpscaler called with no network')
    const { backend, network, image } = this
    return this.upscalerMode === 'bicubic'
      ? new BicubicUpscaler(backend,  network.output, image.h, image.w)
      : new BilinearUpscaler(backend, network.output, image.h, image.w)
  }

  private makeCompositor(): CompositorHandle {
    if (!this.upscaler) throw new Error('makeCompositor called before upscaler exists')
    const { backend, image, upscaler, bgConfig } = this
    switch (bgConfig.mode) {
      case 'solid': return new CompositorSolid(backend, image, upscaler.output, bgConfig.color)
      case 'image': return new CompositorImage(backend, image, upscaler.output, bgConfig.image)
      case 'blur':  return new CompositorBlur(backend,  image, upscaler.output, bgConfig.sigma)
    }
  }
}
