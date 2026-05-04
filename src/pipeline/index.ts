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
  effect:   EffectConfig
  weights:  ArrayBuffer                       // raw .bin (or .f16.bin) — see docs/WEIGHTS_FORMAT.md
  preset?:  PresetName | ManualPreset         // default: 'auto'
  audio?:   AudioMode                          // default: 'passthrough'
  enabled?: boolean                            // default: true
  onReady?: () => void
}

const DEFAULTS = {
  preset:  'auto'         as PresetName,
  audio:   'passthrough'  as AudioMode,
  enabled: true,
}

// Output canvas size for non-transfer-capture topologies. TODO: surface as
// a PipelineOption so callers can pick their output resolution.
const DEFAULT_CANVAS = { w: 1280, h: 720 }

export class Pipeline implements PromiseLike<Pipeline> {
  readonly stream: MediaStream
  readonly ready:  Promise<this>

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

    this.ready = new Promise<this>((resolve, reject) => {
      this.controller.addPersistentListener('ready', () => {
        opts.onReady?.()
        resolve(this)
      })
      this.controller.addPersistentListener('error', (info) => {
        console.error('[pipeline] worker error:', info.message)
        if (!info.recoverable) reject(new Error(info.message))
      })
    })

    // Send init. Transferables (ports) move ownership to the worker.
    // Fire-and-forget here — success is reported via the 'ready' event,
    // failure via the 'error' event (both wired above).
    const initData: InitData = {
      topology,
      preset:     opts.preset,
      effect:     opts.effect,
      enabled:    opts.enabled,
      weights:    opts.weights,
      backend:    'auto',
      dtype:      'f16',
      inputPort:  inputSetup.port,
      outputPort: outputSetup.port,
    }
    void this.controller.sendMessage('init', initData, [
      ...inputSetup.transferList,
      ...outputSetup.transferList,
    ])
  }

  // Thenable — `await pipeline` resolves to this once ready.
  then<T1 = this, T2 = never>(
      onFulfilled?: ((value: this) => T1 | PromiseLike<T1>) | null,
      onRejected?:  ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return this.ready.then(onFulfilled, onRejected)
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
