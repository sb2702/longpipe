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

// What compositeTo() can render: an effect (BackgroundConfig), raw passthrough
// (input image straight to the target — disabled/boot), or fg-passthrough (the
// EFFECT-CHAIN tail straight to the target — background 'none' with an active
// effect chain, e.g. touch-up without a virtual background). Effect specs also
// read the chain tail as their foreground: the chain composes UPSTREAM of the
// one terminal compositor per target.
export type CompositeSpec = BackgroundConfig | { mode: 'passthrough' } | { mode: 'fg-passthrough' }

export interface NetworkLike {
  readonly output: Tensor
  run(): void
}

// A Tensor→Tensor effect stage (e.g. FaceTouchupStage): consumes the previous
// foreground image tensor (bound at construction) and produces the next. The
// chain runs once per frame (runEffects) before any composite; the terminal
// compositor(s) bind the chain tail as their foreground.
export interface EffectStage {
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

  // Ordered Tensor→Tensor effect chain (empty = foreground is the raw display
  // image). Stages bind their input tensor at construction; the renderer
  // rebuilds the chain when effects change (setEffectChain clears the
  // compositor cache — every effect compositor binds the chain tail).
  private effectChain: EffectStage[] = []

  // The GEOMETRIC slot: auto-reframe. `viewRect` is a 1×1×4 (cx, cy, size,
  // moving) tensor owned by the renderer (it holds the camera state across
  // frames); RenderOp only APPLIES it. Two sample ops, rebuilt whenever the
  // chain tail or the alpha source changes:
  //   viewFg    — samples the effect-chain tail (or the raw image)
  //   viewAlpha — samples the upscaled alpha
  // Deliberately NOT applied to the background: solid/image backgrounds stay put
  // while the subject zooms (a virtual backdrop shouldn't move), and blur is
  // derived from the foreground so it follows for free. The reframe sits AFTER
  // the effect chain because touch-up draws its mesh at full-frame landmark
  // coords — reframing upstream would put every face effect in the wrong place.
  private viewRect:  Tensor | null = null
  private viewFg:    { readonly output: Tensor; run(): void } | null = null
  private viewAlpha: { readonly output: Tensor; run(): void } | null = null

  // Deduplicates the per-frame display-image upload: setSource marks dirty,
  // refreshDisplayInput uploads once, later calls are no-ops until the next
  // setSource. Lets the renderer refresh early (effect stages read the image
  // before compositing) without double uploads.
  private displayDirty = false

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
    this.rebuildView()          // viewAlpha binds upscaler.output
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
    this.displayDirty = true
  }

  setUpscaler(mode: UpscalerMode): void {
    if (mode === this.upscalerMode) return
    this.upscalerMode = mode
    if (this.network) {
      this.upscaler = this.makeUpscaler()
      this.rebuildView()
      this.compositors.clear()   // every effect compositor binds upscaler.output
    }
  }

  setBackground(config: BackgroundConfig): void {
    // Stored only — compositeTo() rebuilds the 'main' compositor lazily when it
    // sees the spec change (next runDisplay()).
    this.bgConfig = config
  }

  // The raw display image tensor (output-canvas res). Effect stages bind this
  // (or an earlier stage's output) as their input.
  get displayImage(): Tensor {
    return this.image
  }

  // Current foreground: the effect-chain tail (or raw image), reframed if a view
  // is active.
  private fgImage(): Tensor {
    return this.viewFg ? this.viewFg.output : this.chainTail()
  }

  private chainTail(): Tensor {
    return this.effectChain.length ? this.effectChain[this.effectChain.length - 1].output : this.image
  }

  // Current alpha: the upscaler output, reframed if a view is active.
  private alphaImage(): Tensor {
    if (!this.upscaler) throw new Error('RenderOp.alphaImage called before attachNetwork')
    return this.viewAlpha ? this.viewAlpha.output : this.upscaler.output
  }

  // Attach (or clear) the reframe view. `rect` is the renderer-owned camera-state
  // tensor; passing null removes the reframe entirely.
  setViewTransform(rect: Tensor | null): void {
    this.viewRect = rect
    this.rebuildView()
    this.compositors.clear()
  }

  private rebuildView(): void {
    this.viewFg = null
    this.viewAlpha = null
    if (!this.viewRect) return
    this.viewFg = this.backend.ops.Reframe(this.chainTail(), this.viewRect)
    if (this.upscaler) this.viewAlpha = this.backend.ops.Reframe(this.upscaler.output, this.viewRect)
  }

  // Replace the effect chain. Invalidates all cached compositors — effect and
  // fg-passthrough compositors bind the (possibly new) chain tail.
  setEffectChain(stages: EffectStage[]): void {
    this.effectChain = stages
    this.rebuildView()          // viewFg binds the chain tail
    this.compositors.clear()
  }

  // Run the chain once per frame, after refreshDisplayInput and before any
  // composite. No-op when the chain is empty.
  runEffects(): void {
    for (const s of this.effectChain) s.run()
    // Every frame, not just inference frames: the rect eases continuously, so the
    // sample must be re-applied even when the alpha tensor didn't change.
    this.viewFg?.run()
    this.viewAlpha?.run()
  }

  run(): void {
    this.runModel()
    this.runDisplay()
  }

  // Cheap, runs every frame: refresh display tensor, run the effect chain,
  // composite the 'main' target with whatever's currently in the alpha tensor
  // (which persists between model runs).
  runDisplay(): void {
    this.refreshDisplayInput()
    this.runEffects()
    this.compositeMain('effect')
  }

  // Effect-chain output straight to the canvas — background 'none' with an
  // active chain (e.g. touch-up on, no virtual background).
  runFgPassthrough(): void {
    this.refreshDisplayInput()
    this.runEffects()
    this.compositeMain('fg')
  }

  // True passthrough: writes the raw input image directly to the canvas. Used
  // by the renderer when disabled OR while booting (no network attached
  // yet). setSource() is still required (display input needs a fresh
  // frame); model + alpha + effect pipeline is skipped entirely.
  runPassthrough(): void {
    this.refreshDisplayInput()
    this.compositeMain('raw')
  }

  // Composite the 'main' target: the stored effect config, the effect-chain
  // tail (fg), or the raw image (raw). Does NOT refresh the display input —
  // used by the renderer's multi-target frame where refreshDisplayInput() is
  // called once and the WebGL preview path needs main composited LAST (after
  // the preview snapshot) so the output adapter captures main content.
  compositeMain(mode: 'effect' | 'fg' | 'raw'): void {
    const spec: CompositeSpec = mode === 'effect' ? this.bgConfig
      : mode === 'fg' ? { mode: 'fg-passthrough' }
      : { mode: 'passthrough' }
    this.compositeTo('main', spec)
  }

  // Refresh the full-resolution display image from the staged source. Run ONCE
  // per frame before any compositeTo() — the image tensor is shared across all
  // targets (main + preview), so re-running it per target would be wasted work.
  refreshDisplayInput(): void {
    if (!this.displayDirty) return
    this.displayInput.run()
    this.displayDirty = false
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

  // Flow-warped skip frame: replace the network alpha with `alpha` (the warped
  // previous inference) and re-run the upscaler, so the SAME composite path then
  // shows the warped result. Keeps the temporal state (flow net, warp, carrier) in
  // the renderer; render_op stays a thin orchestrator. `alpha` must match
  // network.output's shape (canvas-res alpha). Call before the per-frame composite.
  applyAlpha(alpha: Tensor): void {
    if (!this.network || !this.upscaler) throw new Error('RenderOp.applyAlpha called before attachNetwork')
    this.backend.copyTensor(alpha, this.network.output)
    this.upscaler.run()
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
    // Everything below composites the CHAIN TAIL as its foreground — the one
    // terminal compositor per target; effects composed upstream.
    const fg = this.fgImage()
    if (spec.mode === 'fg-passthrough') {
      return backend.presenters.CompositePassthrough(fg, target)
    }
    if (!this.upscaler) throw new Error('RenderOp.compositeTo (effect spec) called before attachNetwork')
    const alpha = this.alphaImage()
    switch (spec.mode) {
      case 'solid': return new CompositorSolid(backend, fg, alpha, spec.color, target)
      case 'image': return new CompositorImage(backend, fg, alpha, spec.image, target)
      case 'blur':  return new CompositorBlur(backend,  fg, alpha, spec.sigma, target)
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
    case 'fg-passthrough': return true
    case 'solid': {
      const ac = (a as { color: [number, number, number] }).color
      return ac[0] === b.color[0] && ac[1] === b.color[1] && ac[2] === b.color[2]
    }
    case 'blur':  return (a as { sigma: number }).sigma === b.sigma
    case 'image': return (a as { image: Tensor }).image === b.image
  }
}
