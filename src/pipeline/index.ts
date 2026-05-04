// Pipeline — public-facing class. Construct with (inputStream, options);
// the Pipeline owns all internal plumbing (worker, transport setup, control
// channel). Wait for init via `await pipeline.ready` or the `onReady`
// option callback; `pipeline.stream` is available immediately and emits
// passthrough until ready resolves.
//
// The class itself stays a thin coordinator: each step in the constructor
// delegates to a submodule (topology selection, transport setup, audio
// passthrough, worker spawn). See docs/PIPELINE.md.

import type { ManualPreset, PresetName, ModelName } from './presets'
import { PRESETS } from './presets'
import type { EffectConfig, BackgroundConfig } from './effects'
import type { AudioMode } from './audio'
import type { InitData } from './messages'
import type { Topology } from './topology'
import { selectTopology } from './topology'
import { buildOutputStream } from './audio'
import { WorkerController } from './worker_controller'
import { setupPostMessageInput }    from './transports/input_postmessage'
import { setupBitmapShuttleOutput } from './transports/output_bitmap_shuttle'

export interface PipelineOptions {
  effect:          EffectConfig
  preset?:         PresetName | ManualPreset    // default: 'balanced' ('auto' maps until autotune lands)
  weightsBaseUrl?: string                       // default: DEFAULT_WEIGHTS_BASE_URL
  audio?:          AudioMode                    // default: 'passthrough'
  enabled?:        boolean                      // default: true
  onReady?:        () => void
}

// Public CDN where Longpipe hosts its own model weights. Versioned in the
// path so SDK upgrades that change weight shapes can move to a new prefix
// without breaking older SDKs in the wild.
const DEFAULT_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/v/0.0.1/'

const DEFAULTS = {
  preset:         'balanced'                  as PresetName,
  weightsBaseUrl: DEFAULT_WEIGHTS_BASE_URL,
  audio:          'passthrough'               as AudioMode,
  enabled:        true,
}

// Output canvas size for non-transfer-capture topologies. TODO: surface as
// a PipelineOption so callers can pick their output resolution.
const DEFAULT_CANVAS = { w: 1280, h: 720 }

// Adaptive controller knobs. TODO: surface as PipelineOptions when we
// learn what callers actually want to tune.
const ADAPTIVE_INTERVAL_MS    = 2000   // how often to poll getStats
const ADAPTIVE_COOLDOWN_MS    = 10000  // min gap between swaps
const ADAPTIVE_SOURCE_FPS     = 30     // assumed source rate for budget calc
const ADAPTIVE_MIN_FPS        = 15     // fps below this for OVERSHOOT_MS → downgrade
const ADAPTIVE_OVERSHOOT_MS   = 5000
const ADAPTIVE_HEADROOM_FRAC  = 0.3    // modelMs / frame budget below this for HEADROOM_MS → upgrade
const ADAPTIVE_HEADROOM_MS    = 15000  // upgrade is more conservative than downgrade

// Build the full weights URL for a resolved preset. Convention matches
// the training pipeline's binary export naming (model_${name}.bin).
function weightsUrlFor(baseUrl: string, model: string): string {
  return `${baseUrl.replace(/\/$/, '')}/model_${model}.bin`
}

export class Pipeline implements PromiseLike<Pipeline> {
  readonly stream: MediaStream
  // Promise<void> rather than Promise<this> on purpose: resolving a Promise
  // with a thenable value triggers the spec's "follow" behavior — it tries
  // to settle by calling the thenable's then(). Pipeline IS a thenable
  // (see then() below), and its then() delegates back to this.ready —
  // which is the same Promise being resolved. That's a cycle and the
  // promise hangs forever. Resolving with void breaks the cycle. Callers
  // who want the Pipeline as the awaited value use `await pipeline`
  // (which goes through then() and resolves with `this`).
  readonly ready:  Promise<void>

  private controller:    WorkerController
  private worker:        Worker
  private inputCleanup:  () => void
  private outputCleanup: () => void

  // Adaptive controller state — only set if caller passed preset:'auto'
  // (explicit preset choice is respected, never overridden).
  private adaptiveTimer:    ReturnType<typeof setInterval> | null = null
  private adaptiveBackend:  'webgpu' | 'webgl' | null = null
  private adaptiveBaseUrl:  string | null = null
  private currentPresetIdx: number = -1
  private lastSwapAt:       number = 0
  private overshootStart:   number = 0    // first time fps < threshold; reset when fps recovers
  private headroomStart:    number = 0    // first time modelMs comfortable; reset when modelMs spikes

  constructor(inputStream: MediaStream, options: PipelineOptions) {
    const opts = { ...DEFAULTS, ...options }

    // v0.1: only the universal path (rvfc-postmessage in + bitmap-shuttle
    // out) has main-side helpers. Topology selection runs for forward
    // compat but its result is ignored until other paths' main sides land.
    const _topologyForLater = selectTopology()
    void _topologyForLater
    const topology: Topology = { input: 'rvfc-postmessage', output: 'bitmap-shuttle' }

    const inputSetup  = setupPostMessageInput(inputStream)
    const outputSetup = setupBitmapShuttleOutput(DEFAULT_CANVAS)
    this.inputCleanup  = inputSetup.cleanup
    this.outputCleanup = outputSetup.cleanup

    // Output MediaStream available synchronously. Until the worker emits
    // 'ready' on first processed frame, the captureStream-fed track will
    // simply emit nothing (consumer will see a frozen / black frame).
    // Audio passthrough wires the input's audio tracks if requested.
    this.stream = buildOutputStream(outputSetup.videoTrack, inputStream, opts.audio)

    // Worker spawn — `new URL(..., import.meta.url)` is the standard ESM
    // worker pattern; bundlers (vite, webpack, rollup) handle it.
    this.worker     = new Worker(new URL('./worker/index.ts', import.meta.url), { type: 'module' })
    this.controller = new WorkerController(this.worker)

    this.ready = new Promise<void>((resolve, reject) => {
      this.controller.addPersistentListener('ready', () => {
        console.log('[longpipe/pipeline] ready handler invoked; resolving .ready')
        opts.onReady?.()
        resolve()
      })
      this.controller.addPersistentListener('error', (info) => {
        console.error('[longpipe/pipeline] worker error:', info.message)
        if (!info.recoverable) reject(new Error(info.message))
      })
    })

    // Two-phase init: send 'init' (no weights) → await InitResponse with
    // resolved preset → fetch weights from baseUrl based on resolved
    // preset → send 'startRender' with weights → worker constructs
    // renderer + starts pipe → 'ready' event fires on first frame.
    const initData: InitData = {
      topology,
      preset:     opts.preset,
      effect:     opts.effect,
      enabled:    opts.enabled,
      backend:    'auto',
      dtype:      'f16',
      inputPort:  inputSetup.port,
      outputPort: outputSetup.port,
    }
    const transferList: Transferable[] = [
      ...inputSetup.transferList,
      ...outputSetup.transferList,
    ]

    void this.bootstrap(initData, transferList, opts.weightsBaseUrl)
  }

  // Async second half of construction: init handshake → fetch weights →
  // startRender. Errors are emitted via 'error' event (which the ready
  // promise listens to and rejects on).
  private async bootstrap(
    initData:       InitData,
    transferList:   Transferable[],
    weightsBaseUrl: string,
  ): Promise<void> {
    try {
      console.log('[longpipe/pipeline] sending init…')
      const initRes = await this.controller.sendMessage('init', initData, transferList)
      console.log('[longpipe/pipeline] init resolved:', initRes)

      const url = weightsUrlFor(weightsBaseUrl, initRes.resolvedPreset.model)
      console.log('[longpipe/pipeline] fetching weights:', url)
      const r = await fetch(url)
      if (!r.ok) throw new Error(`weights fetch failed: ${r.status} ${url}`)
      const weights = await r.arrayBuffer()
      console.log('[longpipe/pipeline] weights bytes:', weights.byteLength)

      console.log('[longpipe/pipeline] sending startRender…')
      await this.controller.sendMessage('startRender', { weights }, [weights])
      console.log('[longpipe/pipeline] startRender resolved; awaiting first frame')

      // Wire adaptive controller — only when caller used 'auto'. Explicit
      // preset choices are respected and never auto-overridden.
      if (initData.preset === 'auto') {
        this.startAdaptive(initRes.resolvedBackend, initRes.resolvedPreset.model, weightsBaseUrl)
      }
    } catch (err) {
      console.error('[longpipe/pipeline] bootstrap failed:', err)
      this.controller['handleMessage'].call(this.controller, {
        data: { request_id: 'error', res: { message: (err as Error).message ?? String(err), recoverable: false } },
      } as MessageEvent)
    }
  }

  // Thenable — `await pipeline` resolves to `this` once ready. Wraps
  // ready (Promise<void>) so the awaited value is the Pipeline instance.
  then<T1 = this, T2 = never>(
      onFulfilled?: ((value: this) => T1 | PromiseLike<T1>) | null,
      onRejected?:  ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return this.ready.then(
      () => onFulfilled ? onFulfilled(this) : (this as unknown as T1),
      onRejected,
    )
  }

  setEffect(e: EffectConfig): void {
    void this.controller.sendMessage('setEffect', e)
  }

  setBackground(c: BackgroundConfig): void {
    this.setEffect({ effect: 'background', config: c })
  }

  setPreset(p: PresetName | ManualPreset, weights?: ArrayBuffer): void {
    void this.controller.sendMessage('setPreset', { preset: p, weights })
  }

  setEnabled(on: boolean): void {
    void this.controller.sendMessage('setEnabled', { enabled: on })
  }

  // Async stats snapshot from the worker. Returned object includes rolling
  // FPS / model time / display time / current preset / etc. — see
  // RendererStats for the full shape. Cheap to call frequently (a single
  // postMessage round-trip; renderer just reads its already-tracked
  // counters).
  async getStats() {
    return this.controller.sendMessage('getStats', {} as Record<string, never>)
  }

  // ── Adaptive controller ────────────────────────────────────────────────
  // Polls getStats() every ADAPTIVE_INTERVAL_MS; downgrades when fps is
  // consistently below ADAPTIVE_MIN_FPS for ADAPTIVE_OVERSHOOT_MS;
  // upgrades (WebGPU only) when modelMs is consistently below
  // (frameBudget × ADAPTIVE_HEADROOM_FRAC) for ADAPTIVE_HEADROOM_MS. After
  // any swap, COOLDOWN_MS before considering another. Skips polls when
  // the document is hidden so background-tab throttling doesn't trigger
  // bogus downgrades.

  private startAdaptive(backend: 'webgpu' | 'webgl', initialModel: ModelName, baseUrl: string): void {
    this.adaptiveBackend  = backend
    this.adaptiveBaseUrl  = baseUrl
    this.currentPresetIdx = PRESETS.findIndex(p => p.model === initialModel)
    this.lastSwapAt       = performance.now()
    console.log(`[longpipe/adaptive] started; backend=${backend} initial=${initialModel} idx=${this.currentPresetIdx}`)
    this.adaptiveTimer = setInterval(() => {
      this.adaptiveTick().catch(err => console.warn('[longpipe/adaptive] tick failed:', err))
    }, ADAPTIVE_INTERVAL_MS)
  }

  private async adaptiveTick(): Promise<void> {
    if (typeof document !== 'undefined' && document.hidden) return
    const now = performance.now()
    if (now - this.lastSwapAt < ADAPTIVE_COOLDOWN_MS) return

    const stats = await this.getStats()

    // Downgrade — works on both backends, FPS-based.
    if (stats.fps < ADAPTIVE_MIN_FPS) {
      if (this.overshootStart === 0) this.overshootStart = now
      if (now - this.overshootStart >= ADAPTIVE_OVERSHOOT_MS) {
        this.tryDowngrade(stats.fps)
      }
    } else {
      this.overshootStart = 0
    }

    // Upgrade — WebGPU only, modelMs-based. Need real GPU timing which
    // WebGL's stats don't provide (sync overhead too high to sample).
    if (this.adaptiveBackend === 'webgpu' && stats.modelMs > 0) {
      const budgetMs = (1000 / ADAPTIVE_SOURCE_FPS) * ADAPTIVE_HEADROOM_FRAC
      if (stats.modelMs < budgetMs) {
        if (this.headroomStart === 0) this.headroomStart = now
        if (now - this.headroomStart >= ADAPTIVE_HEADROOM_MS) {
          this.tryUpgrade(stats.modelMs, budgetMs)
        }
      } else {
        this.headroomStart = 0
      }
    }
  }

  private tryDowngrade(fps: number): void {
    if (this.currentPresetIdx <= 0) return  // already at the floor
    const next = this.currentPresetIdx - 1
    console.log(`[longpipe/adaptive] downgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (fps=${fps})`)
    void this.swapToPreset(next)
  }

  private tryUpgrade(modelMs: number, budgetMs: number): void {
    if (this.currentPresetIdx >= PRESETS.length - 1) return  // already at the ceiling
    const next = this.currentPresetIdx + 1
    console.log(`[longpipe/adaptive] upgrade ${PRESETS[this.currentPresetIdx].model} → ${PRESETS[next].model} (modelMs=${modelMs.toFixed(1)} < ${budgetMs.toFixed(1)})`)
    void this.swapToPreset(next)
  }

  private async swapToPreset(idx: number): Promise<void> {
    const preset = PRESETS[idx]
    this.lastSwapAt     = performance.now()    // start cooldown immediately so next tick won't re-trigger
    this.overshootStart = 0
    this.headroomStart  = 0
    try {
      const url = weightsUrlFor(this.adaptiveBaseUrl!, preset.model)
      const r   = await fetch(url)
      if (!r.ok) throw new Error(`weights fetch failed: ${r.status} ${url}`)
      const weights = await r.arrayBuffer()
      await this.controller.sendMessage('setPreset', { preset: preset.model as PresetName, weights }, [weights])
      this.currentPresetIdx = idx
      console.log(`[longpipe/adaptive] swap to ${preset.model} done`)
    } catch (err) {
      console.warn('[longpipe/adaptive] swap failed:', err)
      // Leave currentPresetIdx unchanged; cooldown still applies so we
      // don't immediately retry.
    }
  }

  destroy(): void {
    if (this.adaptiveTimer) clearInterval(this.adaptiveTimer)
    this.adaptiveTimer = null
    void this.controller.sendMessage('destroy', {} as Record<string, never>)
    this.controller.terminate()
    this.inputCleanup()
    this.outputCleanup()
  }
}

// Public type re-exports
export type { PipelineOptions as Options }
export type { EffectConfig, BackgroundConfig }                from './effects'
export type { PresetName, ManualPreset, ModelName }           from './presets'
export type { AudioMode }                                     from './audio'
