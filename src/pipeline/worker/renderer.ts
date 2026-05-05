// Transport-agnostic core. Wraps RenderOp from sdk/src/model/render_op.ts
// with frame-skipping logic. Adapters call process(frame) per input frame;
// renderer decides whether to run the model this frame.

import type { Backend, Tensor, InputOp } from '~/model/backend.ts'
// Note: this file builds the GPU compute chain (network + RenderOp). The
// Streams API pipe chain (inputReadable → transform → outputWritable) is
// wired in worker/index.ts handleInit() where transport adapters are
// composed with this renderer.
import type { ModelWeights } from '~/model/weights.ts'
import { RenderOp, type BackgroundConfig as RenderOpBackgroundConfig } from '~/model/render_op.ts'
import { loadWeightsFromBinary } from '~/utils/loadWeights.ts'
import { EfficientNetLiteMattingLarge }   from '~/model/networks/efficientnetlite_matting_large.ts'
import { EfficientNetLiteMattingCompact } from '~/model/networks/efficientnetlite_matting_compact.ts'
import { EfficientNetLiteMattingSmall }   from '~/model/networks/efficientnetlite_matting_small.ts'
import { EfficientNetLiteMattingXL }      from '~/model/networks/efficientnetlite_matting_xl.ts'
import type { ManualPreset, ModelName } from '../presets'
import type { Background } from '../background'
import type { RendererStats } from '../messages'
import type { Topology } from '../topology'

export interface RendererOptions {
  backend:     Backend
  backendKind: 'webgpu' | 'webgl'    // gates GPU-time sampling (WebGL's fence polling is too heavy for per-frame use)
  canvas:      OffscreenCanvas
  background:  Background
  enabled:     boolean
  topology:    Topology              // reported in stats for debug
}

// Renderer is constructed in boot mode (no preset / no weights / no network).
// Pipeline starts processing frames immediately in passthrough — that lets
// the consumer see live video during autotune + weight fetch (1-3s on
// modest hardware). Call setPreset(preset, weights) once the choice is
// resolved + weights are loaded; the renderer attaches the network in
// place and switches to effect mode mid-pipe (no pipe restart needed).

// Only sample real GPU-time every Nth model run, and only on WebGPU.
// WebGPU sync is one native promise per call (cheap); WebGL sync is a
// setTimeout(1ms) polling loop per call (multiple in flight = many
// concurrent loops). Sampling 1-in-5 keeps the adaptive logic fed without
// adding measurement overhead that would itself perturb the measurement.
const MODEL_TIMING_SAMPLE_INTERVAL = 5

interface NetworkLike { readonly output: Tensor; run(): void }
type NetworkCtor = new (backend: Backend, input: Tensor, w: ModelWeights) => NetworkLike

// xs / small2 / medium share architectures with `small` / `large` (per
// docs/MODEL_PLAN.md) — only the input resolution and dtype differ, which
// flow in via the input Tensor and backend respectively, not the class.
const NETWORK_CTORS: Partial<Record<ModelName, NetworkCtor>> = {
  xxs:     EfficientNetLiteMattingSmall   as unknown as NetworkCtor,
  xs:      EfficientNetLiteMattingSmall   as unknown as NetworkCtor,
  small:   EfficientNetLiteMattingSmall   as unknown as NetworkCtor,
  compact: EfficientNetLiteMattingCompact as unknown as NetworkCtor,
  medium:  EfficientNetLiteMattingLarge   as unknown as NetworkCtor,
  large:   EfficientNetLiteMattingLarge   as unknown as NetworkCtor,
  xl:      EfficientNetLiteMattingXL      as unknown as NetworkCtor,
}

const STATS_WINDOW_MS = 1000

export class Renderer {
  readonly canvas: OffscreenCanvas

  private readonly backend:     Backend
  private readonly backendKind: 'webgpu' | 'webgl'
  private readonly topology:    Topology
  // null until the first setPreset call. Stats use a 'boot' placeholder
  // while null; process() runs passthrough.
  private preset:  ManualPreset | null = null
  private enabled: boolean
  // Current background tracked here so setPreset can rebuild the RenderOp
  // with the same background (otherwise an adaptive swap would flash to
  // a hardcoded fallback).
  private currentBackground: Background
  private modelTimingCounter: number = 0

  private renderOp:     RenderOp
  // Persistent Input op for 'image' background mode; rebuilt when image changes.
  private bgImageInput: InputOp | null = null

  // Frame-skipping: counter-based per preset.skipFrames. Model runs when
  // counter hits 0; counter is reset to skipFrames after each run and
  // decremented every other frame. Counter starts at 0 so first frame
  // always runs the model (warm alpha tensor for compositor).
  private skipCounter: number = 0

  // Stats — rolling windows trimmed to STATS_WINDOW_MS. Duration arrays
  // are stored as { ts, ms } pairs so the trim-by-time logic can compare
  // against a timestamp; previous shape (raw ms numbers) was getting
  // wiped every poll because trimRecent treated the durations as
  // timestamps and dropped them all.
  private framesRenderedAt: number[]                       = []
  private modelRunSamples:   { ts: number; ms: number }[]   = []
  private displayRunSamples: { ts: number; ms: number }[]   = []
  private skippedCount:      number                          = 0

  constructor(opts: RendererOptions) {
    this.backend           = opts.backend
    this.backendKind       = opts.backendKind
    this.canvas            = opts.canvas
    this.enabled           = opts.enabled
    this.currentBackground = opts.background
    this.topology          = opts.topology

    // Boot mode — RenderOp builds only displayInput + passthroughCompositor.
    // The network is attached later via setPreset() once weights arrive.
    this.renderOp = new RenderOp(this.backend)
  }

  // Per-frame entry point. Always cheap (display). Sometimes expensive (model).
  process(frame: VideoFrame): void {
    this.renderOp.setSource(frame)

    if (!this.enabled || this.currentBackground.kind === 'none' || !this.renderOp.hasNetwork()) {
      // True passthrough at the GPU level: input image written directly to
      // canvas, no model, no alpha math. Output stream / MSTG / shuttle
      // path picks up the unmodified frame as if no pipeline existed.
      // Triggered by:
      //   - `enabled: false`        (whole SDK paused)
      //   - `background: 'none'`    (no bg-related effect)
      //   - no network attached yet (worker still booting — autotune /
      //     weight fetch in flight; RenderOp is in boot mode)
      this.renderOp.runPassthrough()
      this.framesRenderedAt.push(performance.now())
      this.trim(this.framesRenderedAt)
      return
    }

    if (this.shouldRunModel()) {
      const t = performance.now()
      this.renderOp.runModel()
      // GPU-time sampling: WebGPU only (cheap native promise per sync;
      // WebGL's setTimeout-polling fence is too heavy at per-frame rate).
      // Even on WebGPU, sample 1 in N runs to keep concurrent in-flight
      // syncs bounded. The adaptive controller only needs a few samples
      // per second to drive its decisions.
      if (this.backendKind === 'webgpu' && this.modelTimingCounter % MODEL_TIMING_SAMPLE_INTERVAL === 0) {
        this.backend.sync().then(() => {
          this.modelRunSamples.push({ ts: performance.now(), ms: performance.now() - t })
          this.trimSamples(this.modelRunSamples)
        }).catch(() => { /* sync errors during shutdown are expected */ })
      }
      this.modelTimingCounter++
    } else {
      this.skippedCount++
    }

    const td = performance.now()
    this.renderOp.runDisplay()
    this.displayRunSamples.push({ ts: performance.now(), ms: performance.now() - td })
    this.trimSamples(this.displayRunSamples)

    this.framesRenderedAt.push(performance.now())
    this.trim(this.framesRenderedAt)
  }

  private shouldRunModel(): boolean {
    if (this.skipCounter === 0) {
      this.skipCounter = this.preset?.skipFrames ?? 0
      return true
    }
    this.skipCounter--
    return false
  }

  setBackground(bg: Background): void {
    this.currentBackground = bg
    // RenderOp.setBackground is a no-op when no network is attached
    // (config gets stored, applied on next setPreset).
    this.renderOp.setBackground(this.translateBackground(bg))
  }

  setEnabled(on: boolean): void {
    this.enabled = on
  }

  // Attach the network — first call wires it from boot mode, subsequent
  // calls swap presets at runtime (used by adaptive controller). The
  // RenderOp instance is reused: only the network-dependent pieces
  // (network, networkInput, upscaler, compositor) are rebuilt.
  setPreset(preset: ManualPreset, weightsBuf: ArrayBuffer): void {
    const Ctor = NETWORK_CTORS[preset.model]
    if (!Ctor) throw new Error(`renderer: model '${preset.model}' not implemented in src/model/networks/`)

    const weights      = loadWeightsFromBinary(weightsBuf)
    const networkInput = this.backend.ops.Input(preset.resolution.h, preset.resolution.w)
    const network      = new Ctor(this.backend, networkInput.output, weights)

    this.renderOp.attachNetwork(network, networkInput, {
      upscaler:   'bilinear',
      background: this.translateBackground(this.currentBackground),
    })
    this.preset      = preset
    this.skipCounter = 0
  }

  getStats(): RendererStats {
    const now = performance.now()
    this.trimRecent(this.framesRenderedAt, now)
    this.trimSamplesAt(this.modelRunSamples,   now)
    this.trimSamplesAt(this.displayRunSamples, now)
    return {
      // window is 1s, so length = events-per-second
      fps:        this.framesRenderedAt.length,
      modelFps:   this.modelRunSamples.length,
      modelMs:    median(this.modelRunSamples.map(s => s.ms)),
      displayMs:  median(this.displayRunSamples.map(s => s.ms)),
      skipped:    this.skippedCount,
      preset:     this.preset?.model ?? 'boot',
      skipFrames: this.preset?.skipFrames ?? 0,
      enabled:    this.enabled,
      inputPath:  this.topology.input,
      outputPath: this.topology.output,
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private translateBackground(bg: Background): RenderOpBackgroundConfig {
    switch (bg.kind) {
      case 'none':
        // RenderOp always builds a compositor; we pass a stub solid-black
        // config that never runs because process() short-circuits to
        // runPassthrough() when kind === 'none'.
        return { mode: 'solid', color: [0, 0, 0] }
      case 'color':
        return { mode: 'solid', color: bg.rgb }
      case 'blur':
        return { mode: 'blur', sigma: bg.sigma }
      case 'image':
        if (!this.bgImageInput) {
          this.bgImageInput = this.backend.ops.Input(this.canvas.height, this.canvas.width)
        }
        this.bgImageInput.setSource(bg.bitmap)
        this.bgImageInput.run()
        return { mode: 'image', image: this.bgImageInput.output }
    }
  }

  // Hard-cap helpers: bound buffer sizes so stale data can't grow without
  // bound between getStats() calls.
  private trim(arr: number[]): void {
    if (arr.length > 240) arr.splice(0, arr.length - 240)
  }
  private trimSamples(arr: { ts: number; ms: number }[]): void {
    if (arr.length > 240) arr.splice(0, arr.length - 240)
  }

  // Trim by time. Compares against the entry's ts field for sample arrays
  // and against the value itself for raw timestamp arrays.
  private trimRecent(arr: number[], now: number): void {
    const cutoff = now - STATS_WINDOW_MS
    let i = 0
    while (i < arr.length && arr[i] < cutoff) i++
    if (i > 0) arr.splice(0, i)
  }
  private trimSamplesAt(arr: { ts: number; ms: number }[], now: number): void {
    const cutoff = now - STATS_WINDOW_MS
    let i = 0
    while (i < arr.length && arr[i].ts < cutoff) i++
    if (i > 0) arr.splice(0, i)
  }
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const sorted = xs.slice().sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}
