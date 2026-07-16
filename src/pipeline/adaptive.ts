// Adaptive preset switching. Polls the renderer's stats, downgrades when
// FPS drops below threshold, and (WebGPU only) upgrades when modelMs has
// consistent headroom. WebGL never upgrades because its per-frame
// modelMs sampling cost is too high — see renderer.ts MODEL_TIMING_SAMPLE_INTERVAL
// for the gating, and adaptive.ts in this file for the consequences.
//
// Decoupled from Pipeline via callbacks (getStats, swapPreset) so this
// module doesn't depend on WorkerController or Pipeline internals.

import type { ModelName, PresetName } from './presets'
import { PRESETS, ManualPreset } from './presets'
import type { RendererStats, PipelineError } from './messages'
import { createLogger } from './debug'

const log = createLogger('adaptive')

// Tunables. Hardcoded for v0.1; surface as PipelineOptions later if
// callers actually want to tune.
const POLL_INTERVAL_MS = 2000
const COOLDOWN_MS      = 10000   // min gap between any two swaps
const SOURCE_FPS       = 30      // assumed source rate for budget calc
const MIN_FPS          = 20      // fps below this for OVERSHOOT_MS → downgrade
const OVERSHOOT_MS     = 5000
const HEADROOM_FRAC    = 0.3     // modelMs / frame budget below this for HEADROOM_MS → upgrade
const HEADROOM_MS      = 15000   // upgrade is more conservative than downgrade

export interface AdaptiveOptions {
  backendKind:     'webgpu' | 'webgl'
  initialModel:    ModelName
  // Fetch weights for a model (resolves f16/f32 pack + fallback in the caller).
  fetchWeights:    (model: ModelName) => Promise<ArrayBuffer>
  getStats:        () => Promise<RendererStats>
  // Caller is responsible for the actual setPreset wire-up; adaptive
  // just hands over fresh weights and the new preset name.
  swapPreset:      (preset: PresetName | ManualPreset, weights: ArrayBuffer) => Promise<void>
  // Surfaced when a swap attempt fails (weights fetch 404, etc.).
  // Adaptive keeps running on the previous preset; cooldown still
  // applies so we don't immediately retry. Recoverable.
  onError?:        (err: PipelineError) => void
}

export class AdaptiveController {
  private timer:            ReturnType<typeof setInterval> | null = null
  private currentPresetIdx: number
  private lastSwapAt:       number = 0
  private overshootStart:   number = 0    // first time fps < threshold; reset on recovery
  private headroomStart:    number = 0    // first time modelMs comfortable; reset on spike

  constructor(private opts: AdaptiveOptions) {
    this.currentPresetIdx = PRESETS.findIndex(p => p.model === opts.initialModel)
    this.lastSwapAt       = performance.now()
  }

  start(): void {
    log(`started; backend=${this.opts.backendKind} initial=${this.opts.initialModel} idx=${this.currentPresetIdx}`)
    this.timer = setInterval(() => {
      this.tick().catch(err => log('tick failed:', err))
    }, POLL_INTERVAL_MS)
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = null
  }

  private async tick(): Promise<void> {
    // Background tabs throttle setInterval/raf; modelMs/fps numbers are
    // unreliable. Skip the tick rather than risk a spurious downgrade.
    if (typeof document !== 'undefined' && document.hidden) return

    const now = performance.now()
    if (now - this.lastSwapAt < COOLDOWN_MS) return

    const stats = await this.opts.getStats()

    // Downgrade — both backends, FPS-based.
    if (stats.fps < MIN_FPS) {
      if (this.overshootStart === 0) this.overshootStart = now
      if (now - this.overshootStart >= OVERSHOOT_MS) this.tryDowngrade(stats.fps)
    } else {
      this.overshootStart = 0
    }

    // Upgrade — WebGPU only, modelMs-based. WebGL stats don't have GPU
    // timing because per-frame fence sync is too heavy.
    if (this.opts.backendKind === 'webgpu' && stats.modelMs > 0) {
      const budgetMs = (1000 / SOURCE_FPS) * HEADROOM_FRAC
      if (stats.modelMs < budgetMs) {
        if (this.headroomStart === 0) this.headroomStart = now
        if (now - this.headroomStart >= HEADROOM_MS) this.tryUpgrade(stats.modelMs, budgetMs)
      } else {
        this.headroomStart = 0
      }
    }
  }

  private tryDowngrade(fps: number): void {
    if (this.currentPresetIdx <= 0) return
    const next = this.currentPresetIdx - 1
    log(`downgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (fps=${fps})`)
    void this.swap(next)
  }

  private tryUpgrade(modelMs: number, budgetMs: number): void {
    // Adaptive follows the same policy as autotune: never upgrade INTO xl —
    // it's the explicit opt-in tier ('quality' / manual preset only).
    if (PRESETS[this.currentPresetIdx + 1]?.model === 'xl') return
    if (this.currentPresetIdx >= PRESETS.length - 1) return
    const next = this.currentPresetIdx + 1
    log(`upgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (modelMs=${modelMs.toFixed(1)} < ${budgetMs.toFixed(1)})`)
    void this.swap(next)
  }

  private async swap(idx: number): Promise<void> {
    const preset = PRESETS[idx]
    // Start cooldown immediately (before the async fetch) so the next
    // poll can't trigger another swap while this one is in flight.
    this.lastSwapAt     = performance.now()
    this.overshootStart = 0
    this.headroomStart  = 0
    try {
      const weights = await this.opts.fetchWeights(preset.model)
      // Pass the full ManualPreset: bare model names ('large') are NOT
      // PresetNames (those are fast/balanced/quality) and the worker's
      // resolveNamedPreset rejects them — the old `preset.model as PresetName`
      // cast made every adaptive swap fail with "unknown preset".
      await this.opts.swapPreset(preset, weights)
      this.currentPresetIdx = idx
      log(`swap to ${preset.model} done`)
    } catch (err) {
      log('swap failed:', err)
      // Leave currentPresetIdx unchanged; cooldown still applies so we
      // don't immediately retry.
      this.opts.onError?.({
        message:     `adaptive preset swap failed: ${(err as Error).message ?? String(err)}`,
        source:      'adaptive',
        recoverable: true,
        cause:       err,
      })
    }
  }
}
