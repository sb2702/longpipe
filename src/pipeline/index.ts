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
import { WorkerController } from './worker_controller'

export interface PipelineOptions {
  effect:   EffectConfig
  preset?:  PresetName | ManualPreset       // default: 'auto'
  audio?:   AudioMode                        // default: 'passthrough'
  enabled?: boolean                          // default: true
  onReady?: () => void
}

const DEFAULTS = {
  preset:  'auto' as PresetName,
  audio:   'passthrough' as AudioMode,
  enabled: true,
}

export class Pipeline implements PromiseLike<Pipeline> {
  readonly stream: MediaStream
  readonly ready:  Promise<this>

  private controller!: WorkerController

  constructor(_inputStream: MediaStream, _options: PipelineOptions) {
    const _opts = { ...DEFAULTS, ..._options }
    void _opts
    // TODO — high-level orchestration only; each step delegates to a submodule:
    //   1. const topology   = selectTopology()
    //   2. const inputSetup  = setupInputTransport(topology, _inputStream)        // → transports/input_*.ts
    //   3. const outputSetup = setupOutputTransport(topology)                      // → transports/output_*.ts
    //   4. this.stream      = buildOutputStream(outputSetup.track, _inputStream, _opts.audio)
    //   5. const worker     = new Worker(new URL('./worker/index.ts', import.meta.url), { type: 'module' })
    //      this.controller  = new WorkerController(worker)
    //   6. this.ready       = new Promise<void>(r =>
    //        this.controller.addPersistentListener('ready', () => r())
    //      ).then(() => { _opts.onReady?.(); return this })
    //   7. this.controller.sendMessage('init', { ...initData }, [...transferList])
    this.stream = new MediaStream()                       // placeholder
    this.ready  = Promise.reject(new Error('Pipeline not yet implemented'))
  }

  // Thenable — enables `await new Pipeline(...)` to wait for ready.
  then<T1 = this, T2 = never>(
      onFulfilled?: ((value: this) => T1 | PromiseLike<T1>) | null,
      onRejected?:  ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return this.ready.then(onFulfilled, onRejected)
  }

  setEffect(_e: EffectConfig): void {
    // TODO: this.controller.sendMessage('setEffect', _e)
  }

  setBackground(c: BackgroundConfig): void {
    this.setEffect({ effect: 'background', config: c })
  }

  setPreset(_p: PresetName | ManualPreset): void {
    // TODO: fetch weights if needed; this.controller.sendMessage('setPreset', ...)
  }

  setEnabled(_on: boolean): void {
    // TODO: this.controller.sendMessage('setEnabled', { enabled: _on })
  }

  destroy(): void {
    this.controller?.terminate()
  }
}

// Public type re-exports
export type { PipelineOptions as Options }
export type { EffectConfig, BackgroundConfig }                from './effects'
export type { PresetName, ManualPreset, ModelName }           from './presets'
export type { AudioMode }                                     from './audio'
