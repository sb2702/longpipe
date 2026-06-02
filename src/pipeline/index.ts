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
import { AdaptiveController } from './adaptive'
import type { BackgroundInput, Background } from './background'
import { normalizeBackground } from './background'
import type { AudioMode } from './audio'
import type { InitData, PipelineError } from './messages'
import { selectTopology } from './topology'
import { buildOutputStream } from './audio'
import { WorkerController } from './worker_controller'
import { setupInput }  from './setup_input'
import { setupOutput } from './setup_output'
import { WORKER_SOURCE } from './worker_inline'
import { createLogger, setDebug } from './debug'

const log = createLogger('pipeline')

// Spawn the pipeline worker. In published builds, WORKER_SOURCE is the
// bundled worker code injected by tsup's inline-worker plugin; we wrap
// that in a Blob URL so the worker is same-origin no matter where the SDK
// loaded from. In dev (vite serving src/), WORKER_SOURCE is empty and we
// fall back to URL-based load — same-origin in dev so it works fine.
function createWorker(): Worker {
  if (WORKER_SOURCE) {
    const blob = new Blob([WORKER_SOURCE], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    const worker = new Worker(url, { type: 'module' })
    // The worker spec consumes the URL synchronously during construction,
    // so we can revoke immediately to free the Blob without affecting the
    // already-spawned worker. Skip if you ever see init failures here.
    URL.revokeObjectURL(url)
    return worker
  }
  return new Worker(new URL('./worker/index.ts', import.meta.url), { type: 'module' })
}

export interface PipelineOptions {
  // Background effect. Wide input surface (string keyword, URL, ImageBitmap,
  // <img> element, structured object) — see BackgroundInput in background.ts.
  // Resolved on the main thread by normalizeBackground() before init.
  // Default: 'none' (passthrough).
  background?:     BackgroundInput
  preset?:         PresetName | ManualPreset    // default: 'auto' (worker microbenches, picks best fit)
  weightsBaseUrl?: string                       // default: DEFAULT_WEIGHTS_BASE_URL
  audio?:          AudioMode                    // default: 'passthrough'
  enabled?:        boolean                      // default: true
  // Output canvas dimensions. Default: matches the input video track's
  // intrinsic size (preserves aspect ratio + avoids pointless rescale).
  // Falls back to 1280×720 if the track hasn't reported its size yet.
  outputResolution?: { w: number; h: number }
  // Whether the SDK should auto-adjust preset at runtime when fps drops
  // (downgrade) or modelMs has consistent headroom (upgrade, WebGPU only).
  // Only takes effect when `preset: 'auto'` — explicit preset choices
  // are always respected. Default: true.
  adaptive?:       boolean
  onReady?:        () => void
  // Fires for ALL async / runtime errors after the constructor returns:
  //   - Init failures (weights 404, normalizeBackground URL fail, worker
  //     init exception) — also reject `pipeline.ready` for backwards
  //     compatibility, but onError fires too
  //   - GPU context loss (WebGL webglcontextlost / WebGPU device.lost)
  //   - Worker uncaught failures (pipe broken, command threw)
  //   - Adaptive preset swap failures (recoverable: true; pipeline keeps
  //     running on the prior preset)
  //   - Background runtime issues
  //
  // Does NOT fire for synchronous constructor errors (no input video
  // track, transport setup throws, etc.) — those propagate out of
  // `new Pipeline()` directly. Wrap construction in try/catch if you
  // want to handle those too.
  onError?:        (err: PipelineError) => void
  // Enable verbose internal logging (`[longpipe/pipeline] …`, etc) on both
  // the main thread and inside the worker. Default false — production
  // consumers see nothing in console unless they opt in.
  debug?:          boolean
}

// Public CDN where Longpipe hosts its own model weights. Versioned in the
// path so SDK upgrades that change weight shapes can move to a new prefix
// without breaking older SDKs in the wild.
const DEFAULT_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/v/0.0.2/'

const DEFAULTS = {
  background:     'blur'                      as BackgroundInput,
  preset:         'auto'                  as PresetName,
  weightsBaseUrl: DEFAULT_WEIGHTS_BASE_URL,
  audio:          'passthrough'               as AudioMode,
  enabled:        true,
  adaptive:       true,
  debug:          false,
}

// Fallback output canvas size — only used when neither the caller nor the
// input track tells us a size. 720p is a reasonable safe default for video
// call output.
const FALLBACK_CANVAS = { w: 1280, h: 720 }

// Pick the output canvas dimensions: explicit option > input track's
// intrinsic size > fallback. Reading getSettings() can return empty
// width/height if the track hasn't initialized yet (rare — happens for
// captureStream'd <video> elements before metadata loads); we treat
// 0/undefined as "not reported" and fall back.
function pickOutputSize(
  inputStream: MediaStream,
  override?: { w: number; h: number },
): { w: number; h: number } {
  if (override) return override
  const track    = inputStream.getVideoTracks()[0]
  const settings = track?.getSettings() ?? {}
  return {
    w: settings.width  || FALLBACK_CANVAS.w,
    h: settings.height || FALLBACK_CANVAS.h,
  }
}

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

  // Adaptive controller — only set when caller passed preset:'auto' AND
  // adaptive option isn't explicitly disabled.
  private adaptive: AdaptiveController | null = null

  // Cleanup callback for the current background. Set by normalizeBackground
  // for kinds that own resources (currently: video, which holds a hidden
  // <video> element + rVFC loop + MessageChannel). Called when bg is
  // replaced or the pipeline is destroyed.
  private bgCleanup: (() => void) | null = null

  constructor(inputStream: MediaStream, options: PipelineOptions) {
    const opts = { ...DEFAULTS, ...options }
    // Set debug FIRST so any subsequent log() calls in this constructor
    // (and downstream main-thread modules) honor the flag from the start.
    setDebug(opts.debug)

    // Pick the best transport pair for this browser. Each axis is chosen
    // independently — the worker's renderer is identical across all 6
    // combos. See pipeline/topology.ts for the selection logic and
    // docs/PIPELINE.md for the empirical browser matrix.
    const topology = selectTopology()

    const inputSetup  = setupInput(topology.input,   inputStream)
    const outputSize  = pickOutputSize(inputStream, opts.outputResolution)
    const outputSetup = setupOutput(topology.output, outputSize)
    this.inputCleanup  = inputSetup.cleanup
    this.outputCleanup = outputSetup.cleanup

    // Output MediaStream available synchronously. While the worker boots
    // (autotune + weight fetch + first frame can take 1-3s), bitmap-shuttle
    // can pump input frames straight to the output canvas so the consumer
    // sees live video immediately (auto-stops the moment the worker posts
    // its first bitmap). MSTG/transfer-capture write to surfaces main
    // doesn't own, so passthrough is bitmap-shuttle-only — those topologies
    // just emit nothing until the worker is ready.
    // Audio passthrough wires the input's audio tracks if requested.
    this.stream = buildOutputStream(outputSetup.videoTrack, inputStream, opts.audio)
    outputSetup.startPassthrough?.(inputStream)

    // Worker spawn — Blob-URL pattern when WORKER_SOURCE is inlined by the
    // tsup build (published builds), URL-based fallback in dev mode where
    // src/ is served same-origin by vite. Blob URL means consumers can load
    // the SDK from any origin (npm bundler, esm.sh, jsdelivr) without
    // hitting cross-origin worker restrictions.
    this.worker     = createWorker()
    this.controller = new WorkerController(this.worker)

    this.ready = new Promise<void>((resolve, reject) => {
      this.controller.addPersistentListener('ready', () => {
        log('ready handler invoked; resolving .ready')
        opts.onReady?.()
        resolve()
      })
      this.controller.addPersistentListener('error', (info) => {
        console.error(`[longpipe/pipeline] error (${info.source}):`, info.message)
        // Init-phase fatal errors reject the ready promise (the natural
        // surface for construction failures). Errors after ready resolved
        // can't reject it (already settled) but DO go through onError.
        if (!info.recoverable) reject(new Error(info.message))
        opts.onError?.(info)
      })
    })

    // Two-phase init: normalize background (may fetch URLs) → send 'init'
    // (no weights) → await InitResponse with resolved preset → fetch
    // weights from baseUrl based on resolved preset → send 'startRender'
    // with weights → worker constructs renderer + starts pipe → 'ready'
    // event fires on first frame.
    const partialInit = {
      topology,
      preset:     opts.preset,
      enabled:    opts.enabled,
      backend:    'auto' as const,
      dtype:      'f16'  as const,
      canvasSize: outputSize,
      debug:      opts.debug,
      ...inputSetup.initFields,
      ...outputSetup.initFields,
    }
    const transferList: Transferable[] = [
      ...inputSetup.transferList,
      ...outputSetup.transferList,
    ]

    void this.bootstrap(partialInit, opts.background, transferList, opts.weightsBaseUrl, opts.adaptive, opts.onError)
  }

  // Async second half of construction: normalize background → init handshake
  // → fetch weights → startRender. Errors are emitted via 'error' event
  // (which the ready promise listens to and rejects on).
  private async bootstrap(
    partialInit:    Omit<InitData, 'background'>,
    rawBackground:  BackgroundInput,
    transferList:   Transferable[],
    weightsBaseUrl: string,
    adaptive:       boolean,
    onError?:       (err: PipelineError) => void,
  ): Promise<void> {
    try {
      log('normalizing background…')
      const norm = await normalizeBackground(rawBackground)
      this.bgCleanup = norm.cleanup ?? null
      const initData: InitData = { ...partialInit, background: norm.background }
      const initTransferList = [...transferList, ...(norm.transferList ?? [])]

      log('sending init…')
      const initRes = await this.controller.sendMessage('init', initData, initTransferList)
      log('init resolved:', initRes)

      const url = weightsUrlFor(weightsBaseUrl, initRes.resolvedPreset.model)
      log('fetching weights:', url)
      const r = await fetch(url)
      if (!r.ok) throw new Error(`weights fetch failed: ${r.status} ${url}`)
      const weights = await r.arrayBuffer()
      log('weights bytes:', weights.byteLength)

      log('sending startRender…')
      await this.controller.sendMessage('startRender', { weights }, [weights])
      log('startRender resolved; awaiting first frame')

      // Adaptive controller — only when caller used 'auto' AND didn't
      // disable it. Explicit preset choices are respected and never
      // auto-overridden.
      if (initData.preset === 'auto' && adaptive) {
        this.adaptive = new AdaptiveController({
          backendKind:     initRes.resolvedBackend,
          initialModel:    initRes.resolvedPreset.model,
          weightsBaseUrl,
          buildWeightsUrl: weightsUrlFor,
          getStats:        () => this.getStats(),
          swapPreset:      async (preset, weights) => {
            await this.controller.sendMessage('setPreset', { preset, weights }, [weights])
          },
          onError,
        })
        this.adaptive.start()
      }
    } catch (err) {
      console.error('[longpipe/pipeline] bootstrap failed:', err)
      // Inject a synthetic 'error' event so the ready promise rejects
      // through the same listener path as worker-emitted errors. Bracket
      // access to handleMessage is intentional — it's private but this
      // is the cleanest re-entry point.
      this.controller['handleMessage'].call(this.controller, {
        data: { request_id: 'error', res: {
          message:     (err as Error).message ?? String(err),
          source:      'worker',
          recoverable: false,
          cause:       err,
        } },
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

  // Swap the background at runtime. Same wide input surface as construction;
  // returns a Promise that resolves once the new background has been
  // normalized (URL fetched, bitmap decoded, video setup, etc.) and the
  // worker has applied it. Throws on parse errors / failed loads.
  // The previous background's resources (e.g. video element) are cleaned
  // up after the worker confirms the new background is in place — earlier
  // teardown could leave the worker reading from a torn-down port.
  async setBackground(input: BackgroundInput): Promise<void> {
    const norm = await normalizeBackground(input)
    const previousCleanup = this.bgCleanup
    this.bgCleanup = norm.cleanup ?? null
    await this.controller.sendMessage('setBackground', norm.background, norm.transferList)
    previousCleanup?.()
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
    this.adaptive?.stop()
    this.adaptive = null
    this.bgCleanup?.()
    this.bgCleanup = null
    void this.controller.sendMessage('destroy', {} as Record<string, never>)
    this.controller.terminate()
    this.inputCleanup()
    this.outputCleanup()
  }
}

// Public type re-exports
export type { PipelineOptions as Options }
export type {
  BackgroundInput, Background, BlurInput, ColorInput, ImageInput, VideoInput,
}                                                             from './background'
export type { PresetName, ManualPreset, ModelName }           from './presets'
export type { AudioMode }                                     from './audio'
export type { PipelineError, ErrorSource }                    from './messages'
