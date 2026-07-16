// Pipeline — public-facing class. Construct with (inputStream, options);
// the Pipeline owns all internal plumbing (worker, transport setup, control
// channel). Wait for init via `await pipeline.ready` or the `onReady`
// option callback; `pipeline.stream` is available immediately and emits
// passthrough until ready resolves.
//
// The class itself stays a thin coordinator: each step in the constructor
// delegates to a submodule (topology selection, transport setup, audio
// passthrough, worker spawn). See docs/PIPELINE.md.

import type { Dtype, FaceTouchupParams, FaceTouchupStyle } from '~/model/backend'
import type { ManualPreset, PresetName } from './presets'
import { AdaptiveController } from './adaptive'
import type { BackgroundInput, Background } from './background'
import { normalizeBackground } from './background'
import type { AudioInput } from './audio'
import type { InitData, PipelineError } from './messages'
import { selectTopology } from './topology'
import { buildOutputStream, normalizeAudio } from './audio'
import { AudioDenoiser } from '~/audio/denoiser.ts'
import type { DenoiseOptions } from './audio'
import type { AudioStats } from '~/audio/messages.ts'
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
  // Numeric precision preference. Default 'f16' (halves bandwidth + weight
  // downloads; falls back to f32 when the adapter lacks shader-f16). Force
  // 'f32' for maximum fidelity or to A/B-test f16-specific issues.
  dtype?:          Dtype
  // 'passthrough' | 'drop' | 'denoise' | { denoise: DenoiseOptions }.
  // 'denoise' runs input audio through the AudioWorklet denoiser (separate from
  // the video worker). Default: 'passthrough'.
  audio?:          AudioInput
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
  // Face touch-up (UV-space skin smoothing). Presence enables it; runs in
  // PARALLEL with any background effect (one shared encoder pass; the
  // retouched frame feeds the background compositor). Requires tier weights
  // with a face blob + the landmark assets at weightsBaseUrl
  // (model_landmark_mesh.bin, face_topology.json, weight_mask.png).
  touchup?:        TouchupOptions
  reframe?:        ReframeConfig
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

// Developer-facing touch-up options; thresholds/decode internals stay SDK-owned.
export interface TouchupOptions {
  strength?: number             // 0..1 blend, default 0.6
  amount?:   number             // smoothing sigma in atlas px, default 8
  detail?:   number             // freq-sep high-band keep, default 0.35
  style?:    FaceTouchupStyle   // 'freq-sep' (default) | 'bilateral'
}

// Auto-reframe. `true` = every default, tracking. The smoothing pair lives under
// `auto` because that's literally its scope: deadband/ease describe how the frame
// MOVES, and in manual it never moves — it's solved once and frozen.
//   reframe: true                            → track, all defaults
//   reframe: { zoom: 1.6 }                   → track, tighter crop
//   reframe: { auto: { deadband: 0.12 } }    → track, lazier
//   reframe: { auto: false }                 → solve once, then freeze; call
//                                              pipeline.reframe() to re-solve
export type ReframeConfig = boolean | ReframeOptions

export interface ReframeOptions {
  zoom?:    number   // crop = frame / zoom, default 1.35 (relaxed toward 1 as needed)
  gravity?: number   // pull toward the subject, default 0.5 — 1 would centre it exactly
  margin?:  number   // keep-out band around the subject, default 0.04
  auto?:    boolean | ReframeAutoOptions   // default true
}

export interface ReframeAutoOptions {
  deadband?: number   // hold until the target moves this far, default 0.09
  ease?:     number   // per-frame lerp toward the target while moving, default 0.07
}

const TOUCHUP_DEFAULTS = { strength: 0.6, amount: 8, detail: 0.35, style: 'freq-sep' as FaceTouchupStyle }
const TOUCHUP_THRESH = 0.15

// Public CDN where Longpipe hosts its own model weights. Versioned in the
// path so SDK upgrades that change weight shapes can move to a new prefix
// without breaking older SDKs in the wild.
// One versioned path hosts everything — video model weights (model_*.bin) and
// audio denoise assets (rnnoise.wasm / dfn.wasm / dfn_weights.pack). Filenames
// disambiguate; no per-kind subfolder.
const DEFAULT_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/v/0.0.4/'

const DEFAULTS = {
  background:     'blur'                  as BackgroundInput,
  preset:         'auto'                  as PresetName,
  weightsBaseUrl: DEFAULT_WEIGHTS_BASE_URL,
  audio:          'passthrough'           as AudioInput,
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

// Build the full weights URL for a resolved preset. Convention matches the
// training pipeline's binary export naming: model_${name}.bin (f32) and
// model_${name}.f16.bin (f16-packed, ~half the size).
function weightsUrlFor(baseUrl: string, model: string, dtype: Dtype): string {
  const suffix = dtype === 'f16' ? '.f16.bin' : '.bin'
  return `${baseUrl.replace(/\/$/, '')}/model_${model}${suffix}`
}

// Fetch a model's weights, preferring the f16 pack when the backend resolved to
// f16 (≈half the download). Falls back to the f32 .bin if the f16 one isn't
// hosted (e.g. a CDN that only ships f32) — the f16 backend converts on upload,
// and the loader decodes f16 bits correctly either way (see utils/weights.ts).
async function fetchModelWeights(baseUrl: string, model: string, dtype: Dtype): Promise<ArrayBuffer> {
  if (dtype === 'f16') {
    const r = await fetch(weightsUrlFor(baseUrl, model, 'f16'))
    if (r.ok) return r.arrayBuffer()
  }
  const url = weightsUrlFor(baseUrl, model, 'f32')
  const r = await fetch(url)
  if (!r.ok) throw new Error(`weights fetch failed: ${r.status} ${url}`)
  return r.arrayBuffer()
}

// Touch-up static assets: the landmark regressor weights (separate .bin — one
// model shared by all tiers) + the canonical face topology + weight mask.
// Fetched once per pipeline and cached (see setTouchup).
interface TouchupAssets {
  landmarkWeights: ArrayBuffer
  topoCount: number
  topoUv: Float32Array
  topoIdx: Float32Array
  weightMask: ImageBitmap
}

async function fetchTouchupAssets(baseUrl: string, dtype: Dtype): Promise<TouchupAssets> {
  const base = baseUrl.replace(/\/$/, '')
  const lmPromise = (async () => {
    if (dtype === 'f16') {
      const r = await fetch(`${base}/model_landmark_mesh.f16.bin`)
      if (r.ok) return r.arrayBuffer()
    }
    const url = `${base}/model_landmark_mesh.bin`
    const r = await fetch(url)
    if (!r.ok) throw new Error(`touchup: landmark weights fetch failed: ${r.status} ${url}`)
    return r.arrayBuffer()
  })()
  const topoPromise = fetch(`${base}/face_topology.json`).then(r => {
    if (!r.ok) throw new Error(`touchup: face_topology.json fetch failed: ${r.status}`)
    return r.json()
  })
  const maskPromise = fetch(`${base}/weight_mask.png`).then(async r => {
    if (!r.ok) throw new Error(`touchup: weight_mask.png fetch failed: ${r.status}`)
    return createImageBitmap(await r.blob())
  })
  const [landmarkWeights, topo, weightMask] = await Promise.all([lmPromise, topoPromise, maskPromise])
  return {
    landmarkWeights,
    topoCount: topo.count,
    topoUv: new Float32Array(topo.uv),
    topoIdx: new Float32Array(topo.idx),
    weightMask,
  }
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

  // Audio denoiser — only set when audio: 'denoise' (or {denoise}) was passed
  // AND the input stream has an audio track. Independent of the video worker.
  private denoiser: AudioDenoiser | null = null

  // Touch-up asset cache + resolved fetch context (set during bootstrap).
  private touchupAssets:  TouchupAssets | null = null
  private weightsBaseUrl: string = DEFAULT_WEIGHTS_BASE_URL
  private resolvedDtype:  Dtype = 'f32'

  // Cleanup callback for the current background. Set by normalizeBackground
  // for kinds that own resources (currently: video, which holds a hidden
  // <video> element + rVFC loop + MessageChannel). Called when bg is
  // replaced or the pipeline is destroyed.
  private bgCleanup: (() => void) | null = null

  // Same as bgCleanup but for the preview background (setPreview). Cleared on
  // clearPreview / replacement / destroy.
  private previewBgCleanup: (() => void) | null = null

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
    // Audio: passthrough/drop wire (or not) the input tracks; 'denoise' spins up
    // the AudioDenoiser (separate AudioWorklet subsystem) and routes its output
    // track. The denoiser's track is live immediately (passthrough until its
    // worklet loads), so the output stream is whole synchronously. Video `ready`
    // does NOT wait on audio — denoise joins asynchronously.
    const audio = normalizeAudio(opts.audio)
    let denoisedTrack: MediaStreamTrack | undefined
    if (audio.mode === 'denoise') {
      const inputAudioTrack = inputStream.getAudioTracks()[0]
      if (inputAudioTrack) {
        this.denoiser = new AudioDenoiser(inputAudioTrack, {
          model:          audio.denoise?.model ?? 'auto',
          weightsBaseUrl: opts.weightsBaseUrl,
          postFilterBeta: audio.denoise?.postFilterBeta,
          gruLeak:        audio.denoise?.gruLeak,
          enabled:        audio.denoise?.enabled,
          onError:        (message) => opts.onError?.({ message, source: 'audio', recoverable: true }),
        })
        denoisedTrack = this.denoiser.outputTrack
      } else {
        opts.onError?.({ message: 'audio: "denoise" requested but the input stream has no audio track', source: 'audio', recoverable: true })
      }
    }
    this.stream = buildOutputStream(outputSetup.videoTrack, inputStream, audio, denoisedTrack)
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
      dtype:      opts.dtype ?? ('f16' as const),
      canvasSize: outputSize,
      debug:      opts.debug,
      ...inputSetup.initFields,
      ...outputSetup.initFields,
    }
    const transferList: Transferable[] = [
      ...inputSetup.transferList,
      ...outputSetup.transferList,
    ]

    this.weightsBaseUrl = opts.weightsBaseUrl
    void this.bootstrap(partialInit, opts.background, transferList, opts.weightsBaseUrl, opts.adaptive, opts.onError, opts.touchup, opts.reframe)
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
    touchup?:       TouchupOptions,
    reframe?:       ReframeConfig,
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

      log('fetching weights:', initRes.resolvedPreset.model, initRes.resolvedDtype)
      const weights = await fetchModelWeights(weightsBaseUrl, initRes.resolvedPreset.model, initRes.resolvedDtype)
      log('weights bytes:', weights.byteLength)

      this.resolvedDtype = initRes.resolvedDtype

      log('sending startRender…')
      await this.controller.sendMessage('startRender', { weights }, [weights])
      log('startRender resolved; awaiting first frame')

      // Initial touch-up (non-blocking — ready never waits on touch-up assets).
      if (reframe) {
        void this.setReframe(reframe).catch(err => {
          onError?.({ message: `reframe init failed: ${(err as Error).message ?? err}`, source: 'unknown', recoverable: true, cause: err })
        })
      }
      if (touchup) {
        void this.setTouchup(touchup).catch(err => {
          onError?.({ message: `touchup init failed: ${(err as Error).message ?? err}`, source: 'unknown', recoverable: true, cause: err })
        })
      }

      // Adaptive controller — only when caller used 'auto' AND didn't
      // disable it. Explicit preset choices are respected and never
      // auto-overridden.
      if (initData.preset === 'auto' && adaptive) {
        this.adaptive = new AdaptiveController({
          backendKind:     initRes.resolvedBackend,
          initialModel:    initRes.resolvedPreset.model,
          fetchWeights:    (model) => fetchModelWeights(weightsBaseUrl, model, initRes.resolvedDtype),
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

  // Enable / update / disable the face touch-up at runtime. First call fetches
  // the landmark assets from weightsBaseUrl (cached for later param updates);
  // pass null to disable. Runs in parallel with the background effect. Requires
  // the loaded tier weights to carry a face blob (all current tiers do) — if
  // they don't, the worker silently leaves the effect off.
  async setTouchup(opts: TouchupOptions | null): Promise<void> {
    await this.ready
    if (opts === null) {
      await this.controller.sendMessage('setTouchup', null)
      return
    }
    if (!this.touchupAssets)
      this.touchupAssets = await fetchTouchupAssets(this.weightsBaseUrl, this.resolvedDtype)
    const params: FaceTouchupParams = {
      strength: opts.strength ?? TOUCHUP_DEFAULTS.strength,
      amount:   opts.amount   ?? TOUCHUP_DEFAULTS.amount,
      detail:   opts.detail   ?? TOUCHUP_DEFAULTS.detail,
      style:    opts.style    ?? TOUCHUP_DEFAULTS.style,
      thresh:   TOUCHUP_THRESH,
    }
    // Structured-clone (no transfer) — the cached assets stay reusable.
    await this.controller.sendMessage('setTouchup', { ...this.touchupAssets, params })
  }

  // Enable / update / disable auto-reframe at runtime. `true` takes every
  // default; `false` (or null) turns it off. Needs no assets — it rides the face
  // heatmaps the tier already carries — so unlike setTouchup this never fetches.
  async setReframe(cfg: ReframeConfig | null): Promise<void> {
    await this.ready
    const normalized = cfg === true ? {} : (cfg === false || cfg === null ? null : cfg)
    await this.controller.sendMessage('setReframe', normalized)
  }

  // Re-solve the frame now. This is the manual-mode entry point — the method to
  // wire to a "reframe" button. In auto mode the camera already tracks, so this
  // is a no-op there.
  async reframe(): Promise<void> {
    await this.ready
    await this.controller.sendMessage('reframeNow', undefined)
  }

  setPreset(p: PresetName | ManualPreset, weights?: ArrayBuffer): void {
    void this.controller.sendMessage('setPreset', { preset: p, weights })
  }

  // ── Preview ───────────────────────────────────────────────────────────────
  // Render a *candidate* effect to a second canvas while the main outgoing
  // stream keeps its currently-applied effect — for "preview before you apply"
  // UX (browsing a background menu). The network + alpha are shared with the
  // main render (computed once); only the compositor differs. Preview is
  // lower-priority: throttled to `fps` (default 15) and skippable.

  // Attach the preview output canvas — call ONCE. The canvas's control is
  // transferred to the worker (transferControlToOffscreen), so the element
  // can't be drawn to by the page afterward and can't be transferred again.
  // The worker resizes the canvas backing to the output resolution (the
  // compositors don't resample), so just CSS-size the element for display.
  attachPreview(canvas: HTMLCanvasElement): void {
    const offscreen = canvas.transferControlToOffscreen()
    void this.controller.sendMessage('attachPreview', { canvas: offscreen }, [offscreen])
  }

  // Set / update the previewed candidate. `background` is the identical wide
  // surface as construction + setBackground (keyword, URL, ImageBitmap, color,
  // blur, image, video). 'none' previews the no-effect (raw) option. Resolves
  // once the worker has applied it (after any URL/bitmap load).
  async setPreview(input: { background: BackgroundInput; fps?: number }): Promise<void> {
    const norm = await normalizeBackground(input.background)
    const previousCleanup = this.previewBgCleanup
    this.previewBgCleanup = norm.cleanup ?? null
    await this.controller.sendMessage(
      'setPreview',
      { background: norm.background, fps: input.fps },
      norm.transferList,
    )
    previousCleanup?.()
  }

  // Stop previewing. The preview canvas freezes on its last frame (hide it in
  // your UI). To "apply" a previewed effect, call setBackground(sameInput).
  clearPreview(): void {
    void this.controller.sendMessage('clearPreview', {} as Record<string, never>)
    this.previewBgCleanup?.()
    this.previewBgCleanup = null
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

  // Reconfigure the audio denoiser at runtime. `true`/`false` toggles
  // denoise vs. passthrough (cheap); an object updates DFN β / GRU-leak live.
  // Only effective when audio: 'denoise' was set at construction (the
  // AudioContext graph is built there); no-op otherwise. Live model swap is a
  // later addition (re-fetch + re-splice).
  setDenoise(input: boolean | DenoiseOptions): void {
    if (!this.denoiser) return
    if (typeof input === 'boolean') { this.denoiser.setEnabled(input); return }
    if (input.postFilterBeta != null || input.gruLeak != null) {
      this.denoiser.setConfig({ postFilterBeta: input.postFilterBeta, gruLeak: input.gruLeak })
    }
    if (input.enabled != null) this.denoiser.setEnabled(input.enabled)
  }

  // Latest per-hop audio telemetry (p50/p95 ms vs the render-quantum deadline,
  // buffered latency, resolved sample rate). null until the worklet reports, or
  // when denoise isn't active. Synchronous — the worklet pushes stats; we cache.
  getAudioStats(): AudioStats | null {
    return this.denoiser?.getStats() ?? null
  }


  destroy(): void {
    this.adaptive?.stop()
    this.adaptive = null
    this.denoiser?.destroy()
    this.denoiser = null
    this.bgCleanup?.()
    this.bgCleanup = null
    this.previewBgCleanup?.()
    this.previewBgCleanup = null
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
export type { AudioMode, AudioInput, DenoiseOptions }         from './audio'
export type { DenoiseModel, DenoiseTier, DenoiseModelOption } from '~/audio/kernels.ts'
export type { AudioStats }                                    from '~/audio/messages.ts'
export type { PipelineError, ErrorSource }                    from './messages'
