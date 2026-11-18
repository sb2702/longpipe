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
const SOURCE_FPS       = 30      // configured target rate (cap) for budget calc
const MIN_FPS          = 20      // at a 30 fps target: fps below this for OVERSHOOT_MS → downgrade
const MIN_FPS_FRAC     = MIN_FPS / SOURCE_FPS   // …scaled to the actual target (2/3)
const OVERSHOOT_MS     = 5000
const HEADROOM_FRAC    = 0.3     // modelMs / frame budget below this for HEADROOM_MS → upgrade
const HEADROOM_MS      = 15000   // upgrade is more conservative than downgrade

export interface AdaptiveOptions {
  backendKind:     'webgpu' | 'webgl'
  initialModel:    ModelName
  // Measured input (camera) fps, or null when unknown. The controller's
  // target is min(input fps, 30): a 15 fps camera rendering 15 fps is on
  // target, not overloaded. Omitted/null → assume a 30 fps source.
  getInputFps?:    () => number | null
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

  constructor(private opts: AdaptiveOptions, now: number = performance.now()) {
    this.currentPresetIdx = PRESETS.findIndex(p => p.model === opts.initialModel)
    this.lastSwapAt       = now
  }

  get currentModel(): ModelName { return PRESETS[this.currentPresetIdx]?.model ?? this.opts.initialModel }

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

  // One poll. Public (with an injectable clock) so tests can drive it.
  async tick(now: number = performance.now()): Promise<void> {
    // Background tabs throttle setInterval/raf; modelMs/fps numbers are
    // unreliable. Skip the tick rather than risk a spurious downgrade.
    if (typeof document !== 'undefined' && document.hidden) return

    if (now - this.lastSwapAt < COOLDOWN_MS) return

    const stats  = await this.opts.getStats()
    const target = this.targetFps()

    // Downgrade — both backends, FPS-based, relative to the input rate.
    if (stats.fps < target * MIN_FPS_FRAC) {
      if (this.overshootStart === 0) this.overshootStart = now
      if (now - this.overshootStart >= OVERSHOOT_MS) this.tryDowngrade(stats.fps, now)
    } else {
      this.overshootStart = 0
    }

    // Upgrade — WebGPU only, modelMs-based. WebGL stats don't have GPU
    // timing because per-frame fence sync is too heavy.
    if (this.opts.backendKind === 'webgpu' && stats.modelMs > 0) {
      const budgetMs = (1000 / target) * HEADROOM_FRAC
      if (stats.modelMs < budgetMs) {
        if (this.headroomStart === 0) this.headroomStart = now
        if (now - this.headroomStart >= HEADROOM_MS) this.tryUpgrade(stats.modelMs, budgetMs, now)
      } else {
        this.headroomStart = 0
      }
    }
  }

  // Effective target fps: the camera's measured rate, capped at SOURCE_FPS.
  // Tiny measured rates (paused / static screen share) are clamped so we
  // never "target" ~0 fps.
  targetFps(): number {
    const input = this.opts.getInputFps?.() ?? null
    if (input === null || !Number.isFinite(input) || input <= 0) return SOURCE_FPS
    return Math.max(5, Math.min(SOURCE_FPS, input))
  }

  private tryDowngrade(fps: number, now: number): void {
    if (this.currentPresetIdx <= 0) return
    const next = this.currentPresetIdx - 1
    log(`downgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (fps=${fps}, target=${this.targetFps().toFixed(1)})`)
    void this.swap(next, now)
  }

  private tryUpgrade(modelMs: number, budgetMs: number, now: number): void {
    // Adaptive follows the same policy as autotune: never upgrade INTO xl —
    // it's the explicit opt-in tier ('quality' / manual preset only).
    if (PRESETS[this.currentPresetIdx + 1]?.model === 'xl') return
    if (this.currentPresetIdx >= PRESETS.length - 1) return
    const next = this.currentPresetIdx + 1
    log(`upgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (modelMs=${modelMs.toFixed(1)} < ${budgetMs.toFixed(1)})`)
    void this.swap(next, now)
  }

  private async swap(idx: number, now: number): Promise<void> {
    const preset = PRESETS[idx]
    // Start cooldown immediately (before the async fetch) so the next
    // poll can't trigger another swap while this one is in flight.
    this.lastSwapAt     = now
    this.overshootStart = 0
    this.headroomStart  = 0
    try {
      const weights = await this.opts.fetchWeights(preset.model)
      // Pass the full ManualPreset (resolution / skipFrames included).
      await this.opts.swapPreset(preset, weights)
      this.currentPresetIdx = idx
      log(`swap to ${preset.model} done`)
    } catch (err) {
      log('swap failed:', err)
      // Leave currentPresetIdx unchanged; cooldown still applies so we
      // don't immediately retry.
      this.opts.onError?.({
        type:        'adaptive',
        fatal:       false,
        message:     `adaptive preset swap failed: ${(err as Error).message ?? String(err)}`,
        source:      'adaptive',
        recoverable: true,
        cause:       err,
      })
    }
  }
}
