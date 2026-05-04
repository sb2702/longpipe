// Transport-agnostic core. Wraps RenderOp from sdk/src/model/render_op.ts
// with frame-skipping logic. Adapters call process(frame) per input frame;
// renderer decides whether to run the model this frame.

import type { Backend, Tensor, InputOp } from '~/model/backend'
// Note: this file builds the GPU compute chain (network + RenderOp). The
// Streams API pipe chain (inputReadable → transform → outputWritable) is
// wired in worker/index.ts handleInit() where transport adapters are
// composed with this renderer.
import type { ModelWeights } from '~/model/weights'
import { RenderOp, type BackgroundConfig as RenderOpBackgroundConfig } from '~/model/render_op'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { EfficientNetLiteMattingLarge }   from '~/model/networks/efficientnetlite_matting_large'
import { EfficientNetLiteMattingCompact } from '~/model/networks/efficientnetlite_matting_compact'
import { EfficientNetLiteMattingSmall }   from '~/model/networks/efficientnetlite_matting_small'
import { EfficientNetLiteMattingXL }      from '~/model/networks/efficientnetlite_matting_xl'
import type { ManualPreset, ModelName } from '../presets'
import type { EffectConfig, BackgroundConfig } from '../effects'
import type { RendererStats } from '../messages'

export interface RendererOptions {
  backend: Backend
  canvas:  OffscreenCanvas
  preset:  ManualPreset
  weights: ArrayBuffer
  effect:  EffectConfig
  enabled: boolean
}

interface NetworkLike { readonly output: Tensor; run(): void }
type NetworkCtor = new (backend: Backend, input: Tensor, w: ModelWeights) => NetworkLike

// Registry of model name → network constructor. xs / small2 not yet ported
// from training to TS — handleInit / setPreset will throw a clear error if
// the requested preset references one.
const NETWORK_CTORS: Partial<Record<ModelName, NetworkCtor>> = {
  small:   EfficientNetLiteMattingSmall   as unknown as NetworkCtor,
  compact: EfficientNetLiteMattingCompact as unknown as NetworkCtor,
  large:   EfficientNetLiteMattingLarge   as unknown as NetworkCtor,
  xl:      EfficientNetLiteMattingXL      as unknown as NetworkCtor,
}

const STATS_WINDOW_MS = 1000
const DEFAULT_BLUR_SIGMA = 8

export class Renderer {
  readonly canvas: OffscreenCanvas

  private readonly backend: Backend
  private preset:  ManualPreset
  private enabled: boolean

  private renderOp:     RenderOp
  private networkInput: InputOp
  // Persistent Input op for 'image' background mode; rebuilt when image changes.
  private bgImageInput: InputOp | null = null

  // Frame-skipping: model runs at preset.modelFps; display always runs.
  // VideoFrame.timestamp is in microseconds.
  private lastModelRunUs: number = -Infinity

  // Stats — rolling windows trimmed to STATS_WINDOW_MS
  private framesRenderedAt: number[] = []
  private modelRunsMs:      number[] = []
  private skippedCount:     number   = 0

  constructor(opts: RendererOptions) {
    this.backend = opts.backend
    this.canvas  = opts.canvas
    this.preset  = opts.preset
    this.enabled = opts.enabled

    const weights      = loadWeightsFromBinary(opts.weights)
    const { renderOp, networkInput } = this.buildRenderChain(opts.preset, weights, opts.effect)
    this.renderOp     = renderOp
    this.networkInput = networkInput
  }

  // Per-frame entry point. Always cheap (display). Sometimes expensive (model).
  process(frame: VideoFrame): void {
    if (!this.enabled) {
      // True passthrough is handled at the MediaStream level by the Pipeline
      // (input track routed to output without going through the worker).
      // If a frame still arrives here, we no-op so we don't burn GPU.
      return
    }

    this.renderOp.setSource(frame)

    const tsUs = frame.timestamp ?? performance.now() * 1000
    if (this.shouldRunModel(tsUs)) {
      const t = performance.now()
      this.renderOp.runModel()
      this.modelRunsMs.push(performance.now() - t)
      this.trim(this.modelRunsMs)
    } else {
      this.skippedCount++
    }

    this.renderOp.runDisplay()

    this.framesRenderedAt.push(performance.now())
    this.trim(this.framesRenderedAt)
  }

  private shouldRunModel(timestampUs: number): boolean {
    const intervalUs = 1_000_000 / this.preset.modelFps
    if (timestampUs - this.lastModelRunUs >= intervalUs) {
      this.lastModelRunUs = timestampUs
      return true
    }
    return false
  }

  setEffect(config: EffectConfig): void {
    if (config.effect !== 'background') return    // only background in v0.1
    this.renderOp.setBackground(this.translateBackground(config.config))
  }

  setEnabled(on: boolean): void {
    this.enabled = on
  }

  // Runtime preset swap (used by adaptive controller in v0.2). Renderer
  // is designed to support this from day one — no constructor-frozen state.
  setPreset(preset: ManualPreset, weightsBuf: ArrayBuffer): void {
    const weights = loadWeightsFromBinary(weightsBuf)
    // Reuse the current background as-is; translate from the renderOp's
    // current config if needed. For now we re-derive from the most recent
    // effect by passing through a 'solid black' baseline; setEffect is
    // expected to be called immediately after if the caller wants a
    // specific background. Keeps the swap cheap and atomic.
    const fallbackEffect: EffectConfig = { effect: 'background', config: { color: [0, 0, 0] } }
    const { renderOp, networkInput } = this.buildRenderChain(preset, weights, fallbackEffect)
    this.renderOp     = renderOp
    this.networkInput = networkInput
    this.preset       = preset
    this.lastModelRunUs = -Infinity
  }

  getStats(): RendererStats {
    const now = performance.now()
    this.trimRecent(this.framesRenderedAt, now)
    this.trimRecent(this.modelRunsMs,      now)
    return {
      fps:     this.framesRenderedAt.length,        // window is 1s, so length = fps
      modelMs: median(this.modelRunsMs),
      skipped: this.skippedCount,
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private buildRenderChain(
    preset:  ManualPreset,
    weights: ModelWeights,
    effect:  EffectConfig,
  ): { renderOp: RenderOp; networkInput: InputOp } {
    const Ctor = NETWORK_CTORS[preset.model]
    if (!Ctor) throw new Error(`renderer: model '${preset.model}' not implemented in src/model/networks/`)

    const networkInput = this.backend.ops.Input(preset.resolution.h, preset.resolution.w)
    const network      = new Ctor(this.backend, networkInput.output, weights)

    const renderOp = new RenderOp(this.backend, network, networkInput, {
      upscaler:   'bilinear',
      background: this.translateBackground(effect.effect === 'background' ? effect.config : { color: [0, 0, 0] }),
    })
    return { renderOp, networkInput }
  }

  private translateBackground(c: BackgroundConfig): RenderOpBackgroundConfig {
    if ('color' in c) return { mode: 'solid', color: c.color }
    if ('blur'  in c) {
      const sigma = c.blur === true ? DEFAULT_BLUR_SIGMA : c.blur.sigma
      return { mode: 'blur', sigma }
    }
    // image — ingest bitmap/frame into a canvas-sized tensor.
    if (!this.bgImageInput) {
      this.bgImageInput = this.backend.ops.Input(this.canvas.height, this.canvas.width)
    }
    this.bgImageInput.setSource(c.image)
    this.bgImageInput.run()
    return { mode: 'image', image: this.bgImageInput.output }
  }

  // Drop entries older than STATS_WINDOW_MS. Cheap O(n); arrays stay small.
  private trim(arr: number[]): void {
    if (arr.length > 240) arr.splice(0, arr.length - 240)   // hard cap
  }

  private trimRecent(arr: number[], now: number): void {
    const cutoff = now - STATS_WINDOW_MS
    let i = 0
    while (i < arr.length && arr[i] < cutoff) i++
    if (i > 0) arr.splice(0, i)
  }
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const sorted = xs.slice().sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}
