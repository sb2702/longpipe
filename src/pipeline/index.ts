// Pipeline — public-facing class. Construct with (inputStream, options);
// the Pipeline owns all internal plumbing (worker, transport setup, control
// channel). Wait for init via `await pipeline.ready` or the `onReady`
// option callback; `pipeline.stream` is available immediately and emits
// passthrough until ready resolves.
//
// The class itself stays a thin coordinator: each step in the constructor
// delegates to a submodule (topology selection, transport setup, audio
// passthrough, worker spawn). See docs/PIPELINE.md.

import type { ManualPreset, PresetName } from './presets'
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

  destroy(): void {
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
