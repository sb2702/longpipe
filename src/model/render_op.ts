import type { Backend, Tensor, InputOp, ImageSource, RenderTarget } from '~/model/backend.ts'
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

// What compositeTo() can render: an effect (BackgroundConfig) or raw
// passthrough (input image straight to the target, no alpha/bg). The renderer
// picks passthrough for a surface whose background is 'none' — see the 2×2
// main/preview effect matrix.
export type CompositeSpec = BackgroundConfig | { mode: 'passthrough' }

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

  // Network-dependent pieces. Null until attachNetwork().
  private network:      N | null = null
  private networkInput: InputOp | null = null
  private upscaler:     UpscalerHandle | null = null

  // Stored config — applied lazily when the compositor is (re)built.
  private upscalerMode: UpscalerMode      = 'bilinear'
  // The 'main' target's effect config; runDisplay()/run() composite with it.
  // Preview config is passed explicitly by the renderer to compositeTo().
  private bgConfig:     BackgroundConfig  = { mode: 'solid', color: [0, 0, 0] }

  // Per-target compositor cache, keyed by RenderTarget. compositeTo() builds a
  // compositor on first use of a (target, spec) and reuses it until the spec
  // changes (config swap) or attachNetwork/setUpscaler invalidates the alpha
  // source (which clears the whole cache — every effect compositor binds the
  // old upscaler.output). 'passthrough' entries bind only `image` so they'd
  // survive, but clearing wholesale keeps invalidation simple.
  private compositors = new Map<RenderTarget, { spec: CompositeSpec; handle: CompositorHandle }>()

  constructor(private readonly backend: Backend) {
    const canvas = backend.canvas
    this.displayInput = backend.ops.Input(canvas.height, canvas.width)
    this.image        = this.displayInput.output
  }

  // Wire (or rewire) the network chain. Rebuilds the upscaler and invalidates
  // every cached compositor (they reference the previous upscaler.output).
  attachNetwork(network: N, networkInput: InputOp, options: RenderOptions): void {
    this.network      = network
    this.networkInput = networkInput
    this.upscalerMode = options.upscaler
    this.bgConfig     = options.background
    this.upscaler     = this.makeUpscaler()
    this.compositors.clear()
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
      this.upscaler = this.makeUpscaler()
      this.compositors.clear()   // every effect compositor binds upscaler.output
    }
  }

  setBackground(config: BackgroundConfig): void {
    // Stored only — compositeTo() rebuilds the 'main' compositor lazily when it
    // sees the spec change (next runDisplay()).
    this.bgConfig = config
  }

  run(): void {
    this.runModel()
    this.runDisplay()
  }

  // Cheap, runs every frame: refresh display tensor at canvas resolution,
  // composite the 'main' target with whatever's currently in the alpha tensor
  // (which persists between model runs).
  runDisplay(): void {
    this.refreshDisplayInput()
    this.compositeMain(false)
  }

  // True passthrough: writes the input image directly to the canvas. Used
  // by the renderer when disabled OR while booting (no network attached
  // yet). setSource() is still required (display input needs a fresh
  // frame); model + alpha pipeline is skipped entirely.
  runPassthrough(): void {
    this.refreshDisplayInput()
    this.compositeMain(true)
  }

  // Composite the 'main' target with its stored config (passthrough=false) or
  // raw passthrough (passthrough=true). Does NOT refresh the display input —
  // used by the renderer's multi-target frame where refreshDisplayInput() is
  // called once and the WebGL preview path needs main composited LAST (after
  // the preview snapshot) so the output adapter captures main content.
  compositeMain(passthrough: boolean): void {
    this.compositeTo('main', passthrough ? { mode: 'passthrough' } : this.bgConfig)
  }

  // Refresh the full-resolution display image from the staged source. Run ONCE
  // per frame before any compositeTo() — the image tensor is shared across all
  // targets (main + preview), so re-running it per target would be wasted work.
  refreshDisplayInput(): void {
    this.displayInput.run()
  }

  // Composite the (already-computed, current) alpha over `spec`'s background to
  // `target`. Reuses a cached compositor unless the spec changed. Effect specs
  // require attachNetwork() first (they read upscaler.output); 'passthrough'
  // works any time (reads only the display image). Does NOT refresh the display
  // input — call refreshDisplayInput() once per frame first.
  compositeTo(target: RenderTarget, spec: CompositeSpec): void {
    const cached = this.compositors.get(target)
    if (cached && sameSpec(cached.spec, spec)) {
      cached.handle.run()
      return
    }
    const handle = this.buildCompositor(target, spec)
    this.compositors.set(target, { spec, handle })
    handle.run()
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

  // Build a compositor bound to `target` for `spec`. Passthrough needs only the
  // display image (works pre-network); effect modes read upscaler.output.
  private buildCompositor(target: RenderTarget, spec: CompositeSpec): CompositorHandle {
    const { backend, image } = this
    if (spec.mode === 'passthrough') {
      return backend.presenters.CompositePassthrough(image, target)
    }
    if (!this.upscaler) throw new Error('RenderOp.compositeTo (effect spec) called before attachNetwork')
    const alpha = this.upscaler.output
    switch (spec.mode) {
      case 'solid': return new CompositorSolid(backend, image, alpha, spec.color, target)
      case 'image': return new CompositorImage(backend, image, alpha, spec.image, target)
      case 'blur':  return new CompositorBlur(backend,  image, alpha, spec.sigma, target)
    }
  }
}

// Whether two specs are render-equivalent — i.e. the cached compositor can be
// reused. Tensor identity (not contents) for image bg: a video background
// updates the same tensor in place every frame, so it must NOT trigger a
// rebuild; switching to a different image tensor must.
function sameSpec(a: CompositeSpec, b: CompositeSpec): boolean {
  if (a.mode !== b.mode) return false
  switch (b.mode) {
    case 'passthrough': return true
    case 'solid': {
      const ac = (a as { color: [number, number, number] }).color
      return ac[0] === b.color[0] && ac[1] === b.color[1] && ac[2] === b.color[2]
    }
    case 'blur':  return (a as { sigma: number }).sigma === b.sigma
    case 'image': return (a as { image: Tensor }).image === b.image
  }
}
