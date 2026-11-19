// Pipeline — public-facing class. Construct with (inputStream, options);
// the Pipeline owns all internal plumbing (worker, transport setup, control
// channel). Wait for init via `await pipeline.ready` or the `onReady`
// option callback; `pipeline.stream` is available immediately and emits
// passthrough until ready resolves.
//
// Robustness policy (0.0.19): FAIL OPEN. The output video track always
// carries something — the effect when possible, otherwise the raw camera —
// and every failure reaches `onError` as a typed PipelineError that says
// whether it is fatal. Concretely:
//   - no GPU at startup       → raw passthrough, backend 'none' (non-fatal)
//   - bad background          → previous background (or 'none') kept (non-fatal)
//   - GPU device/context lost → raw passthrough now, then a fresh worker +
//                               backend (WebGL2 if WebGPU won't come back),
//                               effects resume, `onRecovered` fires
//   - worker crash / hang     → same as GPU loss (raw, restart, onRecovered)
//   - camera ended            → output video ends, 'camera-ended' (fatal)
//   - control call fails      → that call rejects (typed), settings unchanged
// Raw passthrough stays on the SAME output track on the mstg (Chromium) and
// bitmap-shuttle (Firefox) paths. On transfer-capture (Safari) the output
// canvas belongs to the worker, so a dead worker means a new track:
// `onTrackChanged` fires and `pipeline.stream` is updated in place.
//
// The class itself stays a coordinator: transports, worker spawn, audio,
// adaptive live in submodules. See docs/PIPELINE.md.

import type { Dtype, FaceTouchupParams, FaceTouchupStyle } from '~/model/backend'
import type { ManualPreset, PresetName } from './presets'
import { resolveNamedPreset } from './presets'
import { AdaptiveController } from './adaptive'
import type { BackgroundInput, Background, NormalizedBackground } from './background'
import { normalizeBackground } from './background'
import type { AudioInput } from './audio'
import type {
  InitData, InitResponse, BackendKind, ActiveBackend, PipelineStats, PipelineStatus,
  ReadyInfo, RendererStats, WorkerErrorEvent,
} from './messages'
import type { PipelineError, PipelineErrorType } from './errors'
import { LongpipeError, isLongpipeError, errorMessage, prefixMessage } from './errors'
import type { Topology } from './topology'
import { selectTopology } from './topology'
import { buildOutputStream, normalizeAudio } from './audio'
import { AudioDenoiser } from '~/audio/denoiser.ts'
import { DEFAULT_WEIGHTS_BASE_URL, DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL } from '~/weights_url.ts'
import { MEDIAPIPE_SELFIE_CONFIG } from '~/model/networks/mediapipe_selfie.ts'
import type { MediaPipeSelfieVariant } from '~/model/networks/mediapipe_selfie.ts'
import { resolveModel, validateModelOptions, requiresLongpipe, isMediaPipe } from './model_choice'
import type { ModelChoice, ResolvedModel } from './model_choice'
import type { DenoiseOptions } from './audio'
import type { AudioStats } from '~/audio/messages.ts'
import { WorkerController } from './worker_controller'
import type { InputSetup } from './setup_input'
import { setupInput }  from './setup_input'
import type { OutputHandle } from './setup_output'
import { setupOutput } from './setup_output'
import { InputRateMeter } from './input_rate'
import { TUNING } from './tuning'
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
  // If it fails to load, the pipeline starts with 'none' and reports a
  // non-fatal 'background-load' error.
  background?:     BackgroundInput
  // Segmentation model. 'longpipe' (default): Longpipe's tiers — all effects,
  // presets/autotune/adaptive. 'mediapipe-selfie': Google's MediaPipe Selfie
  // Segmenter on the same GPU pipeline — background effects only (touchup,
  // reframe and non-'auto' presets throw). See pipeline/model_choice.ts.
  model?:          ModelChoice
  preset?:         PresetName | ManualPreset    // default: 'auto' (worker microbenches, picks best fit)
  // default: DEFAULT_WEIGHTS_BASE_URL. MediaPipe weights default to
  // DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL, but when this is set they're fetched
  // from here too (self-hosters keep every file in one folder).
  weightsBaseUrl?: string
  // Numeric precision preference. Default 'f16' (halves bandwidth + weight
  // downloads; falls back to f32 when the adapter lacks shader-f16). Force
  // 'f32' for maximum fidelity or to A/B-test f16-specific issues.
  dtype?:          Dtype
  // GPU backend preference. Default 'auto' (WebGPU when the browser gives the
  // SDK an adapter, else WebGL2) — leave it there in production. The others
  // exist for testing and A/B checks:
  //   'webgpu' — same as 'auto' (still falls back to WebGL2 if WebGPU can't start)
  //   'webgl2' — never try WebGPU
  //   'none'   — no GPU at all: raw passthrough, backend 'none', a non-fatal
  //              'no-backend' error (exercises the fail-open path)
  // The backend actually used is `pipeline.backend` / `ready` / `getStats()`.
  backend?:        BackendPreference
  // 'passthrough' | 'drop' | 'denoise' | { denoise: DenoiseOptions }.
  // 'denoise' runs input audio through the AudioWorklet denoiser (separate from
  // the video worker). Default: 'passthrough'.
  audio?:          AudioInput
  // default: true. false = raw passthrough, and the model weights aren't
  // fetched until setEnabled(true) (`ready` resolves with preset: null).
  enabled?:        boolean
  // Output canvas dimensions. Default: matches the input video track's
  // intrinsic size (preserves aspect ratio + avoids pointless rescale).
  // Falls back to 1280×720 if the track hasn't reported its size yet.
  outputResolution?: { w: number; h: number }
  // Whether the SDK should auto-adjust preset at runtime when fps drops
  // (downgrade) or modelMs has consistent headroom (upgrade, WebGPU only).
  // Only takes effect when `preset: 'auto'` — explicit preset choices
  // are always respected. The target is the camera's measured frame rate
  // (capped at 30), so a 15 fps camera isn't treated as overload.
  // Default: true.
  adaptive?:       boolean
  // Face touch-up (UV-space skin smoothing). Presence enables it; runs in
  // PARALLEL with any background effect (one shared encoder pass; the
  // retouched frame feeds the background compositor). Requires tier weights
  // with a face blob + the landmark assets at weightsBaseUrl
  // (model_landmark_mesh.bin, face_topology.json, weight_mask.png).
  touchup?:        TouchupOptions
  reframe?:        ReframeConfig
  // Fires once, when the effect is live (network attached) — or, with no
  // usable GPU, once the pipeline has settled into raw passthrough
  // (info.backend === 'none'). Same moment `ready` resolves.
  onReady?:        (info: ReadyInfo) => void
  // Every async failure, as a typed PipelineError (`type`, `fatal`). See
  // errors.ts for the list. Fatal errors before `ready` also reject `ready`;
  // nothing after `ready` (control-call failures included) touches it.
  //
  // Does NOT fire for synchronous constructor errors (no input video
  // track, transport setup throws, etc.) — those propagate out of
  // `new Pipeline()` directly. Wrap construction in try/catch if you
  // want to handle those too.
  onError?:        (err: PipelineError) => void
  // Fires after automatic recovery (GPU backend lost, worker crash/hang):
  // effects are live again on the restarted worker.
  onRecovered?:    (info: RecoveredInfo) => void
  // Fires when the output video track had to be REPLACED (transfer-capture /
  // Safari only, after its worker died: the old track's canvas is gone).
  // `pipeline.stream` is already updated; call `sender.replaceTrack(videoTrack)`
  // on your RTCRtpSender. Never fires on the mstg / bitmap-shuttle paths.
  onTrackChanged?: (change: TrackChange) => void
  // audio: 'denoise' only. Fires when the output AUDIO track was replaced: the
  // denoise AudioContext isn't running (autoplay policy; see 'audio-suspended'),
  // so the output carries the original mic audio on a clone of the mic track
  // (`denoised: false`), and again when it runs (`denoised: true`, back on the
  // denoised track). `pipeline.stream` is already updated; call
  // `sender.replaceTrack(audioTrack)` on your audio RTCRtpSender. Never fires
  // when the context starts normally (call EffectsPipeline.unlockAudio() in
  // your click handler to make sure it does).
  onAudioTrackChanged?: (change: AudioTrackChange) => void
  // Enable verbose internal logging (`[longpipe/pipeline] …`, etc) on both
  // the main thread and inside the worker. Default false — production
  // consumers see nothing in console unless they opt in.
  debug?:          boolean
}

export type BackendPreference = 'auto' | 'webgpu' | 'webgl2' | 'none'

// Public preference → worker init value. After WebGPU has been lost twice the
// pipeline prefers WebGL2 whatever was asked (see recover()).
function workerBackend(pref: BackendPreference | undefined, fallback: 'auto' | 'webgl'): InitData['backend'] {
  if (fallback === 'webgl' && pref !== 'none') return 'webgl'
  switch (pref) {
    case 'webgl2': return 'webgl'
    case 'webgpu': return 'webgpu'
    case 'none':   return 'none'
    default:       return 'auto'
  }
}

export interface RecoveredInfo {
  reason:   'backend-lost' | 'worker-crash'
  backend:  ActiveBackend
  // Restart attempts it took (1 = first try).
  attempts: number
}

export interface TrackChange {
  stream:        MediaStream          // === pipeline.stream
  videoTrack:    MediaStreamTrack     // the new output video track
  previousTrack: MediaStreamTrack     // the old one (frozen; stopped on destroy())
  reason:        PipelineErrorType
}

export interface AudioTrackChange {
  stream:        MediaStream          // === pipeline.stream
  audioTrack:    MediaStreamTrack     // the new output audio track
  previousTrack: MediaStreamTrack     // the old one (stays live; stopped on destroy())
  denoised:      boolean              // false = original mic audio (context not running)
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

// Public CDN default — see src/weights_url.ts.

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
// MediaPipe selfie weights: one fp16 pack per variant (lossless — the source
// model is fp16; the backend converts on upload if it runs f32).
async function fetchMediaPipeWeights(baseUrl: string, variant: MediaPipeSelfieVariant): Promise<ArrayBuffer> {
  const url = `${baseUrl.replace(/\/$/, '')}/${MEDIAPIPE_SELFIE_CONFIG[variant].file}`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`weights fetch failed: ${r.status} ${url}`)
  return r.arrayBuffer()
}

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


// Settings the app asked for. Replayed onto a restarted worker so recovery
// resumes exactly where the pipeline was.
interface DesiredState {
  background: BackgroundInput
  enabled:    boolean
  touchup:    TouchupOptions | null
  reframe:    ReframeConfig | null
}

interface Deferred {
  promise:  Promise<void>
  resolve:  () => void
  reject:   (e: LongpipeError) => void
  settled:  boolean
}

function deferred(): Deferred {
  let resolve!: () => void
  let reject!: (e: LongpipeError) => void
  const d = { settled: false } as Deferred
  d.promise = new Promise<void>((res, rej) => { resolve = res; reject = rej })
  d.promise.catch(() => { /* waiters handle it; never an unhandled rejection */ })
  d.resolve = () => { if (!d.settled) { d.settled = true; resolve() } }
  d.reject  = (e) => { if (!d.settled) { d.settled = true; reject(e) } }
  return d
}

// Mark a returned promise as handled (errors also go to onError), so callers
// that ignore it don't get "Uncaught (in promise)". Awaiting it still throws.
function quiet<T>(p: Promise<T>): Promise<T> {
  p.catch(() => {})
  return p
}

const toPublicBackend = (b: BackendKind | 'none'): ActiveBackend =>
  b === 'webgl' ? 'webgl2' : b

export class Pipeline implements PromiseLike<Pipeline> {
  // The output stream. Its video track is stable for the pipeline's life on
  // the mstg / bitmap-shuttle paths; on transfer-capture it can be replaced
  // once after a worker failure (see onTrackChanged) — this object is
  // updated in place.
  readonly stream: MediaStream
  // Resolves with ReadyInfo once the effect is live, or once a GPU-less
  // pipeline has settled into raw passthrough (backend 'none'). Rejects only
  // on a FATAL failure before that point (weights download, worker that
  // can't start). Later failures never touch it.
  //
  // Not Promise<this>: resolving a Promise with a thenable value triggers
  // the spec's "follow" behavior, and Pipeline IS a thenable whose then()
  // delegates back to this.ready — a cycle that hangs forever. ReadyInfo is
  // a plain object. Callers who want the Pipeline use `await pipeline`.
  readonly ready:  Promise<ReadyInfo>

  private readonly opts: PipelineOptions & Required<Pick<PipelineOptions, 'background' | 'preset' | 'weightsBaseUrl' | 'audio' | 'enabled' | 'adaptive' | 'debug'>>
  private inputStream:   MediaStream
  private topology:      Topology
  private readonly outputSize: { w: number; h: number }
  private output:        OutputHandle
  // Outputs replaced by takeoverOutput(): their frozen tracks stay alive
  // until destroy() so an app that missed onTrackChanged still sends frames.
  private retiredOutputs: OutputHandle[] = []
  // Input wiring of the current worker (one per worker; transferred).
  private input:         InputSetup | null = null
  private pendingInput:  InputSetup | null = null

  private worker:        Worker | null = null
  private controller:    WorkerController | null = null
  // Incremented whenever the current worker is replaced/terminated. Every
  // async step re-checks it, so callbacks from a dead worker are ignored.
  private gen = 0

  private status:        PipelineStatus = 'starting'
  private passthroughError: LongpipeError | null = null
  private readyResolve!: (info: ReadyInfo) => void
  private readyReject!:  (err: Error) => void
  private readySettled = false
  private op:            Deferred = deferred()

  private desired:       DesiredState
  private presetOverride: PresetName | ManualPreset | null = null
  private readonly adaptiveEligible: boolean

  // Resolved on the current worker.
  private activeBackend:  ActiveBackend | null = null
  private backendKind:    BackendKind | null = null
  // The tier chosen for the current worker (autotune / named / setPreset).
  // Its weights are only loaded once `modelLoaded` (deferred while disabled).
  private resolvedPreset: ManualPreset | null = null
  private modelLoaded = false
  private modelLoading: Promise<void> | null = null
  private pendingWeights: ArrayBuffer | null = null
  private resolvedDtype:  Dtype = 'f32'
  private preferBackend:  'auto' | 'webgl' = 'auto'
  private webgpuLosses = 0
  // Backend the CURRENT worker's init resolved to — set as soon as init
  // answers, before weights/startRender, so a GPU lost during startup is
  // attributed to the right backend (backendKind is only set once running).
  private workerBackendKind: BackendKind | null = null
  // Backend requested for the current worker attempt (known before init replies).
  private attemptBackend: InitData['backend'] | null = null

  // Recovery bookkeeping.
  private restarts = 0
  private recoveryReason: 'backend-lost' | 'worker-crash' | null = null
  private recoveryAttempts = 0
  private restartTimer: ReturnType<typeof setTimeout> | null = null
  private healthySince = 0

  // Watchdog.
  private watchdog:       ReturnType<typeof setInterval> | null = null
  private busy = 0
  private lastHeartbeat = 0
  private lastFramesOut = -1
  private lastProgressAt = 0
  private inputAtProgress: number | null = null
  private readonly meter: InputRateMeter
  private unwatchInput: (() => void) | null = null

  // Adaptive controller — only set when caller passed preset:'auto' AND
  // adaptive option isn't explicitly disabled (and no setPreset() since).
  private adaptive: AdaptiveController | null = null

  // Audio denoiser — only set when audio: 'denoise' (or {denoise}) was passed
  // AND the input stream has an audio track. Independent of the video worker.
  private denoiser: AudioDenoiser | null = null

  // Touch-up asset cache + resolved fetch context (set during bootstrap).
  private touchupAssets:  TouchupAssets | null = null
  private weightsBaseUrl: string = DEFAULT_WEIGHTS_BASE_URL
  private modelBaseUrl:   string = DEFAULT_WEIGHTS_BASE_URL

  // Resolved segmentation model (see model_choice.ts). MediaPipe models gate
  // off every Longpipe-only feature (touch-up, reframe, presets, adaptive).
  private readonly model: ResolvedModel
  private readonly modelName: ModelChoice

  // Cleanup callback for the current background (video kind owns a hidden
  // <video> + rVFC loop + MessageChannel). Called when bg is replaced, the
  // worker is restarted, or the pipeline is destroyed.
  private bgCleanup: (() => void) | null = null
  // Same, for the preview background (setPreview).
  private previewBgCleanup: (() => void) | null = null
  private previewAttached = false

  constructor(inputStream: MediaStream, options: PipelineOptions) {
    // Explicit `undefined` (e.g. `weightsBaseUrl: maybeUrl`) means "use the default", not "unset it".
    const defined = Object.fromEntries(Object.entries(options).filter(([, v]) => v !== undefined)) as PipelineOptions
    const opts = { ...DEFAULTS, ...defined }
    this.opts = opts
    // Set debug FIRST so any subsequent log() calls in this constructor
    // (and downstream main-thread modules) honor the flag from the start.
    setDebug(opts.debug)

    // Model-dependent option checks throw synchronously, before any side effect
    // (no worker, no transport). Checked against the caller's raw options so the
    // DEFAULTS preset ('auto') doesn't count as an explicit choice.
    validateModelOptions(options.model, options)
    this.modelName = options.model ?? 'longpipe'
    this.inputStream = inputStream

    // Pick the best transport pair for this browser. Each axis is chosen
    // independently — the worker's renderer is identical across all 6
    // combos. See pipeline/topology.ts for the selection logic and
    // docs/PIPELINE.md for the empirical browser matrix.
    this.topology = { ...(TUNING.forceTopology ?? selectTopology()) }

    // Sync transport setup: throws out of the constructor (no video track on
    // mstp, no rVFC, …). The first worker consumes this input setup.
    this.pendingInput = setupInput(this.topology.input, inputStream)
    this.outputSize   = pickOutputSize(inputStream, opts.outputResolution)
    this.model        = resolveModel(options.model, this.outputSize)
    this.output       = setupOutput(this.topology.output, this.outputSize)

    // Audio: passthrough/drop wire (or not) the input tracks; 'denoise' spins up
    // the AudioDenoiser (separate AudioWorklet subsystem) and routes its output
    // track. The denoiser's track is live immediately (passthrough until its
    // worklet loads), so the output stream is whole synchronously. Video `ready`
    // does NOT wait on audio — denoise joins asynchronously. If the denoiser
    // can't even be constructed, the original audio track is used instead.
    let audio = normalizeAudio(opts.audio)
    let denoisedTrack: MediaStreamTrack | undefined
    if (audio.mode === 'denoise') {
      const inputAudioTrack = inputStream.getAudioTracks()[0]
      if (inputAudioTrack) {
        try {
          this.denoiser = new AudioDenoiser(inputAudioTrack, {
            model:          audio.denoise?.model ?? 'auto',
            weightsBaseUrl: opts.weightsBaseUrl,
            postFilterBeta: audio.denoise?.postFilterBeta,
            gruLeak:        audio.denoise?.gruLeak,
            enabled:        audio.denoise?.enabled,
            onError:        (message) => this.emitError(new LongpipeError('audio', message)),
            onSuspended:    () => this.emitError(new LongpipeError('audio-suspended',
              'audio: the AudioContext is suspended by the autoplay policy, so the output carries the original ' +
              '(not denoised) audio on a new track (onAudioTrackChanged); denoise starts on the next user gesture, ' +
              'or call pipeline.resumeAudio() from one')),
            onOutputTrackChanged: (track, previous, denoised) => this.swapAudioTrack(track, previous, denoised),
          })
          denoisedTrack = this.denoiser.outputTrack
        } catch (err) {
          // e.g. Firefox NotSupportedError on a sample-rate mismatch.
          this.denoiser = null
          audio = { mode: 'passthrough' }
          queueMicrotask(() => this.emitError(new LongpipeError('audio',
            `audio denoise unavailable (${errorMessage(err)}); passing the original audio through`, { cause: err })))
        }
      } else {
        queueMicrotask(() => this.emitError(new LongpipeError('audio', 'audio: "denoise" requested but the input stream has no audio track')))
      }
    }
    this.stream = buildOutputStream(this.output.videoTrack, inputStream, audio, denoisedTrack)

    // Raw passthrough on the output track until the worker's frames take over
    // (mstg: relay worker; bitmap-shuttle: main canvas). transfer-capture
    // can't (its canvas goes to the worker) and stays blank until the worker
    // draws — boot passthrough, or raw 2D frames if there's no GPU.
    this.output.startRaw(inputStream)

    this.desired = {
      background: opts.background,
      enabled:    opts.enabled,
      touchup:    options.touchup ?? null,
      reframe:    options.reframe ?? null,
    }
    this.adaptiveEligible = opts.preset === 'auto' && opts.adaptive && !isMediaPipe(this.model)

    this.ready = new Promise<ReadyInfo>((resolve, reject) => {
      this.readyResolve = resolve
      this.readyReject  = reject
    })

    this.weightsBaseUrl = opts.weightsBaseUrl
    // MediaPipe weights: the caller's weightsBaseUrl if they set one (self-
    // hosting, one flat folder), else their own CDN prefix.
    this.modelBaseUrl = isMediaPipe(this.model) && options.weightsBaseUrl === undefined
      ? DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL
      : opts.weightsBaseUrl

    this.meter = new InputRateMeter(
      () => this.inputStream.getVideoTracks()[0],
      () => this.input?.framesIn ?? this.pendingInput?.framesIn,
    )
    this.watchInputTrack()
    this.watchdog = setInterval(() => this.watchdogTick(), 1000)

    void this.startWorker()
  }

  // The denoiser's output track changed (its AudioContext stopped / started
  // running): swap it in pipeline.stream and tell the app.
  private swapAudioTrack(track: MediaStreamTrack, previous: MediaStreamTrack, denoised: boolean): void {
    if (this.status === 'destroyed') return
    this.stream.removeTrack(previous)
    this.stream.addTrack(track)
    log(`audio output track changed (${denoised ? 'denoised' : 'original audio: AudioContext not running'})`)
    safeCall(() => this.opts.onAudioTrackChanged?.({ stream: this.stream, audioTrack: track, previousTrack: previous, denoised }))
  }

  // Let audio denoise start without a later gesture: creates (and resumes) the
  // AudioContext the next pipeline's denoiser will use. Call it synchronously
  // in the click handler that starts the call, BEFORE awaiting getUserMedia —
  // the browser only allows an AudioContext to start during a user gesture,
  // and that has expired by the time the stream arrives. Harmless otherwise.
  static unlockAudio(): void {
    AudioDenoiser.unlock()
  }

  // ── Public state ──────────────────────────────────────────────────────────

  // Active GPU backend: 'webgpu' | 'webgl2' | 'none' (no usable GPU, or
  // effects unavailable). null until the worker has resolved one.
  get backend(): ActiveBackend | null { return this.activeBackend }

  // Thenable — `await pipeline` resolves to `this` once ready. Wraps
  // ready so the awaited value is the Pipeline instance.
  then<T1 = this, T2 = never>(
      onFulfilled?: ((value: this) => T1 | PromiseLike<T1>) | null,
      onRejected?:  ((reason: unknown) => T2 | PromiseLike<T2>) | null,
  ): PromiseLike<T1 | T2> {
    return this.ready.then(
      () => onFulfilled ? onFulfilled(this) : (this as unknown as T1),
      onRejected,
    )
  }

  // ── Worker lifecycle ──────────────────────────────────────────────────────

  // Spawn a worker and bring it to 'running': normalize background → init
  // (backend + autotune) → fetch weights → startRender → replay settings.
  // Used for the first start AND every restart.
  private async startWorker(): Promise<void> {
    if (this.status === 'destroyed' || this.status === 'ended') return
    const gen = ++this.gen
    const isRestart = this.recoveryReason !== null

    const input = this.pendingInput ?? setupInput(this.topology.input, this.inputStream)
    this.pendingInput = null
    this.input = input

    if (!this.output.reusable && this.worker) {
      // transfer-capture's canvas went to a previous worker.
      this.takeoverOutput(this.recoveryReason ?? 'worker-crash')
    }
    const attach = this.output.attachWorker()

    this.workerBackendKind = null
    const worker = createWorker()
    const controller = new WorkerController(worker, (message, cause) => {
      if (gen === this.gen) this.recover('worker-crash', message, cause)
    })
    this.worker = worker
    this.controller = controller
    controller.addPersistentListener('error',      (ev) => this.onWorkerError(gen, ev))
    controller.addPersistentListener('heartbeat',  (hb) => this.onHeartbeat(gen, hb.framesOut))
    controller.addPersistentListener('inputEnded', () => { if (gen === this.gen) this.checkInputEnded() })
    this.lastHeartbeat = performance.now()

    try {
      log(isRestart ? 'restarting worker…' : 'starting worker…')
      const norm = await this.normalizeDesiredBackground()
      if (gen !== this.gen) { norm.cleanup?.(); return }
      this.bgCleanup?.()
      this.bgCleanup = norm.cleanup ?? null

      const preset: PresetName | ManualPreset = this.presetOverride
        ?? (isRestart && this.resolvedPreset && this.backendKind === this.lastBackendKindFor(this.preferBackend)
          ? this.resolvedPreset
          : this.opts.preset)

      const initData: InitData = {
        topology:    this.topology,
        model:       this.model,
        preset,
        background:  norm.background,
        enabled:     this.desired.enabled,
        backend:     (this.attemptBackend = TUNING.forceBackend ?? workerBackend(this.opts.backend, this.preferBackend)),
        dtype:       this.opts.dtype ?? 'f16',
        canvasSize:  this.outputSize,
        debug:       this.opts.debug,
        heartbeatMs: TUNING.heartbeatMs,
        ...input.initFields,
        ...attach.initFields,
      }
      const transferList: Transferable[] = [
        ...input.transferList,
        ...attach.transferList,
        ...(norm.transferList ?? []),
      ]

      log('sending init…')
      const initRes = await this.busyCall(() =>
        controller.sendMessage('init', initData, transferList, { timeoutMs: TUNING.initTimeoutMs }))
      if (gen !== this.gen) return
      log('init resolved:', initRes)
      this.resolvedDtype = initRes.resolvedDtype
      this.workerBackendKind = initRes.resolvedBackend === 'none' ? null : initRes.resolvedBackend

      if (initRes.resolvedBackend === 'none') {
        this.onNoBackend(initRes, isRestart)
        return
      }

      this.resolvedPreset = initRes.resolvedPreset
      this.modelLoaded    = false
      this.modelLoading   = null
      // Disabled: no weights until setEnabled(true) (the renderer passes frames through).
      if (this.desired.enabled) {
        await this.loadModel(controller, gen)
        if (gen !== this.gen) return
      }

      this.backendKind    = initRes.resolvedBackend
      this.activeBackend  = toPublicBackend(initRes.resolvedBackend)
      this.onRunning(gen)
    } catch (err) {
      const st = this.status as PipelineStatus
      if (gen !== this.gen || st === 'destroyed' || st === 'ended') return
      if (isLongpipeError(err) && (err.type === 'worker-crash' || err.type === 'control-timeout')) {
        // The worker died or hung mid-startup: same path as a crash.
        this.recover('worker-crash', err.message, err, { silent: isRestart })
        return
      }
      const e = new LongpipeError('init-failed', prefixMessage('init failed', errorMessage(err)), { cause: err })
      if (isRestart) {
        // A restart attempt failed: try again (or give up), without a new
        // onError per attempt.
        log('restart attempt failed:', e.message)
        this.recover(this.recoveryReason ?? 'worker-crash', e.message, err, { silent: true })
        return
      }
      console.error('[longpipe/pipeline] bootstrap failed:', err)
      // The worker (if it got a backend) keeps pumping unprocessed frames;
      // effects are off. Fatal: ready rejects.
      this.enterPassthrough(e)
    }
  }

  private lastBackendKindFor(pref: 'auto' | 'webgl'): BackendKind | null {
    // When forcing WebGL after a WebGPU loss the old (WebGPU-tuned) preset may
    // be too heavy — re-run autotune instead of reusing it.
    return pref === 'webgl' ? 'webgl' : this.backendKind
  }

  // Fetch the current model's weights and attach the network (startRender).
  private async loadModel(controller: WorkerController, gen: number): Promise<void> {
    const preset = this.resolvedPreset
    let weights = this.pendingWeights
    this.pendingWeights = null
    if (!weights) {
      weights = isMediaPipe(this.model)
        ? await fetchMediaPipeWeights(this.modelBaseUrl, this.model.variant)
        : await fetchModelWeights(this.modelBaseUrl, preset!.model, this.resolvedDtype)
    }
    if (gen !== this.gen) return
    log('weights bytes:', weights.byteLength)
    log('sending startRender…')
    const buf = weights
    await this.busyCall(() => controller.sendMessage('startRender',
      { weights: buf, preset: preset ?? undefined }, [buf], { timeoutMs: TUNING.initTimeoutMs }))
    if (gen === this.gen) this.modelLoaded = true
  }

  private onRunning(gen: number): void {
    const wasRecovering = this.recoveryReason
    const attempts = this.recoveryAttempts
    this.status = 'running'
    this.passthroughError = null
    this.healthySince = performance.now()
    this.lastHeartbeat = performance.now()
    this.lastProgressAt = performance.now()
    this.inputAtProgress = this.meter.totalFrames()
    this.recoveryReason = null
    this.recoveryAttempts = 0

    this.replayFeatures(gen)
    this.startAdaptive()
    this.op.resolve()

    const info = this.readyInfo()
    if (!this.readySettled) {
      this.readySettled = true
      log('ready:', info)
      safeCall(() => this.opts.onReady?.(info))
      this.readyResolve(info)
    }
    if (wasRecovering) {
      log('recovered after', wasRecovering, 'attempts:', attempts)
      if (this.previewAttached) {
        this.previewAttached = false
        this.emitError(new LongpipeError('control-failed',
          'preview: the preview canvas was lost when the worker restarted; call attachPreview() with a new canvas'))
      }
      safeCall(() => this.opts.onRecovered?.({ reason: wasRecovering, backend: this.activeBackend ?? 'none', attempts }))
    }
  }

  // No usable GPU. Raw passthrough is already on the output (relay / main
  // canvas), or the worker draws raw 2D frames (transfer-capture); if
  // neither, move the output to a main-owned canvas.
  private onNoBackend(initRes: InitResponse, isRestart: boolean): void {
    this.activeBackend = 'none'
    this.backendKind   = null
    if (this.output.path === 'transfer-capture' && !initRes.workerDrawsRaw) {
      this.takeoverOutput('no-backend')
    }
    // Raw comes from main's output path, so the worker reads no input: stop
    // feeding it (rVFC path: a VideoFrame per camera frame into a closed port).
    if (!initRes.workerDrawsRaw) {
      this.input?.cleanup()
      this.input = null
    }
    const detail = initRes.backendError ? ` (${initRes.backendError})` : ''
    const e = isRestart
      ? new LongpipeError('recovery-failed', `recovery failed: no usable GPU backend after ${this.recoveryReason}${detail}; output is the raw camera`)
      : new LongpipeError('no-backend', `no usable GPU backend${detail}; output is the raw camera, effects are off`)
    this.enterPassthrough(e)
  }

  // Effects unavailable; output carries raw/unprocessed frames. Fatal errors
  // reject `ready` if it hasn't settled; non-fatal ('no-backend') resolve it.
  private enterPassthrough(e: LongpipeError): void {
    this.status = 'passthrough'
    this.passthroughError = e
    this.recoveryReason = null
    this.adaptive?.stop()
    this.adaptive = null
    this.emitError(e)
    this.op.reject(e)
    if (!this.readySettled) {
      this.readySettled = true
      if (e.fatal) this.readyReject(e)
      else {
        const info = this.readyInfo()
        safeCall(() => this.opts.onReady?.(info))
        this.readyResolve(info)
      }
    }
  }

  // Worker crash / hang / pipe failure / GPU loss: raw passthrough NOW, then
  // restart the worker with backoff. `silent` = a retry, don't re-report.
  private recover(
    type: 'backend-lost' | 'worker-crash',
    message: string,
    cause?: unknown,
    opts: { silent?: boolean } = {},
  ): void {
    if (this.status === 'destroyed' || this.status === 'ended') return
    log('recover:', type, message)
    if (this.checkInputEnded()) return
    const first = this.recoveryReason === null
    if (first) this.recoveryReason = type
    // The failed worker never reached 'running' (first start-up, or a restart's).
    const duringStartup = this.status === 'starting' || this.status === 'recovering'
    if (!opts.silent) {
      this.emitError(new LongpipeError(type, `${message}; output switched to the raw camera, restarting`, { cause }))
    }

    // Retire the current worker.
    this.gen++
    this.adaptive?.stop()
    this.adaptive = null
    this.controller?.terminate(type === 'backend-lost' ? 'GPU backend lost' : 'worker restarted', TUNING.retireGraceMs)
    this.input?.cleanup()
    this.input = null
    if (this.restartTimer) clearTimeout(this.restartTimer)
    this.restartTimer = null
    this.status = 'recovering'
    if (this.op.settled) this.op = deferred()

    // Raw on the same track if this path allows it; else a new track.
    if (!this.output.startRaw(this.inputStream)) this.takeoverOutput(type)

    // workerBackendKind, not backendKind: a device lost during startup (before
    // startRender finished) must count too, or a GPU that dies on its first
    // frame retries WebGPU until recovery gives up instead of moving to WebGL2.
    // After two WebGPU failures, use WebGL2. What counts:
    //   - every device loss on WebGPU;
    //   - a worker crash/hang only while that worker was still starting up
    //     (e.g. Firefox's worker dying in WebGPU init). A crash of a worker that
    //     was already running says nothing about WebGPU, so it restarts on the
    //     same backend.
    // A crash can arrive before init replies (frames already flow through the
    // boot renderer), so also count attempts that requested something other than WebGL.
    const onWebGPU = this.workerBackendKind === 'webgpu' ||
      (this.workerBackendKind === null && this.attemptBackend !== null && this.attemptBackend !== 'webgl' && this.attemptBackend !== 'none')
    if (onWebGPU && (type === 'backend-lost' || duringStartup)) {
      this.webgpuLosses++
      if (this.webgpuLosses >= 2) this.preferBackend = 'webgl'
    }

    if (this.restarts >= TUNING.maxRestarts) {
      const e = new LongpipeError('recovery-failed',
        `recovery failed after ${this.restarts} restarts (${this.recoveryReason}: ${message}); output stays on the raw camera`, { cause })
      this.controller = null
      this.worker = null
      this.activeBackend = 'none'
      this.enterPassthrough(e)
      return
    }
    const delay = TUNING.restartBackoffMs[Math.min(this.restarts, TUNING.restartBackoffMs.length - 1)] ?? 1000
    this.restarts++
    this.recoveryAttempts++
    log(`restart #${this.restarts} in ${delay} ms (prefer ${this.preferBackend})`)
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null
      void this.startWorker()
    }, delay)
  }

  // transfer-capture only: the worker that owned the output canvas is gone
  // (or can't draw). Move to a bitmap-shuttle output the main thread owns —
  // a NEW track — with raw passthrough, and tell the app.
  private takeoverOutput(reason: PipelineErrorType): void {
    const prev = this.output
    const next = setupOutput('bitmap-shuttle', this.outputSize)
    this.retiredOutputs.push(prev)
    this.output = next
    this.topology = { ...this.topology, output: 'bitmap-shuttle' }
    next.startRaw(this.inputStream)
    this.stream.removeTrack(prev.videoTrack)
    this.stream.addTrack(next.videoTrack)
    log('output moved to bitmap-shuttle (track changed):', reason)
    safeCall(() => this.opts.onTrackChanged?.({
      stream: this.stream, videoTrack: next.videoTrack, previousTrack: prev.videoTrack, reason,
    }))
  }

  private onWorkerError(gen: number, ev: WorkerErrorEvent): void {
    if (gen !== this.gen) return
    if (ev.type === 'backend-lost') { this.recover('backend-lost', ev.message, ev.cause); return }
    if (ev.type === 'worker-crash') { this.recover('worker-crash', ev.message, ev.cause); return }
    this.emitError(new LongpipeError(ev.type, ev.message, { fatal: ev.fatal, source: ev.source, cause: ev.cause }))
  }

  private onHeartbeat(gen: number, framesOut: number): void {
    if (gen !== this.gen) return
    const now = performance.now()
    this.lastHeartbeat = now
    if (framesOut !== this.lastFramesOut) {
      this.lastFramesOut   = framesOut
      this.lastProgressAt  = now
      this.inputAtProgress = this.meter.totalFrames()
    }
  }

  // Heavy worker commands (shader compile / autotune) can block the worker
  // event loop for seconds: pause the watchdog while one is in flight.
  private async busyCall<T>(fn: () => Promise<T>): Promise<T> {
    this.busy++
    try {
      return await fn()
    } finally {
      this.busy--
      this.lastHeartbeat = Math.max(this.lastHeartbeat, performance.now())
      this.lastProgressAt = Math.max(this.lastProgressAt, performance.now())
      this.inputAtProgress = this.meter.totalFrames()
    }
  }

  private watchdogTick(): void {
    if (this.status === 'destroyed') return
    this.meter.sample()
    if (this.checkInputEnded()) return
    if (this.status !== 'running' || this.busy > 0 || !this.controller) return
    const now = performance.now()
    if (now - this.healthySince > TUNING.stableResetMs) {
      this.restarts = 0
    }
    if (now - this.lastHeartbeat > TUNING.watchdogMs) {
      this.recover('worker-crash', `worker unresponsive for ${Math.round(now - this.lastHeartbeat)} ms`)
      return
    }
    // Stall: the camera kept delivering but the worker's output didn't move.
    const inputNow = this.meter.totalFrames()
    if (inputNow !== null && this.inputAtProgress !== null
        && inputNow - this.inputAtProgress >= 5 && now - this.lastProgressAt > TUNING.stallMs) {
      this.recover('worker-crash', `worker output stalled for ${Math.round(now - this.lastProgressAt)} ms while the camera kept delivering`)
    }
  }

  // ── Input (camera) ───────────────────────────────────────────────────────

  private watchInputTrack(): void {
    this.unwatchInput?.()
    const track = this.inputStream.getVideoTracks()[0]
    if (!track) { this.unwatchInput = null; return }
    const onEnded = () => this.checkInputEnded()
    track.addEventListener('ended', onEnded)
    this.unwatchInput = () => track.removeEventListener('ended', onEnded)
  }

  // `ended` fires when the UA ends the track (unplug, permission revoked);
  // an app calling track.stop() fires nothing, so this also runs on a 1 s
  // poll and when the worker sees its input stream close. Returns true if
  // the pipeline is (now) ended.
  private checkInputEnded(): boolean {
    if (this.status === 'ended') return true
    if (this.status === 'destroyed') return false
    const track = this.inputStream.getVideoTracks()[0]
    if (!track || track.readyState !== 'ended') return false
    this.onCameraEnded()
    return true
  }

  private onCameraEnded(): void {
    log('input video track ended')
    this.status = 'ended'
    this.gen++
    if (this.restartTimer) clearTimeout(this.restartTimer)
    this.restartTimer = null
    this.adaptive?.stop()
    this.adaptive = null
    this.controller?.terminate('camera ended', TUNING.retireGraceMs)
    this.controller = null
    this.input?.cleanup()
    this.input = null
    this.pendingInput?.cleanup()
    this.pendingInput = null
    this.output.end()
    for (const o of this.retiredOutputs) o.end()
    const e = new LongpipeError('camera-ended',
      'the input video track ended; the output video track has ended too. Create a new pipeline (or call setInputStream() before stopping the old camera to switch cameras)')
    this.op.reject(e)
    this.emitError(e)
    if (!this.readySettled) { this.readySettled = true; this.readyReject(e) }
  }

  // Switch the input to a new stream (e.g. camera switch) WITHOUT changing the
  // output track. Call it BEFORE stopping the old camera. Only the video track
  // is switched; audio stays as configured at construction.
  setInputStream(stream: MediaStream): Promise<void> {
    return quiet(this.switchInput(stream).catch((err) => { throw this.reportControlError('setInputStream', err) }))
  }

  private async switchInput(stream: MediaStream): Promise<void> {
    const track = stream.getVideoTracks()[0]
    if (!track || track.readyState === 'ended') {
      throw new LongpipeError('control-failed', 'setInputStream: the stream has no live video track')
    }
    if (this.status === 'destroyed' || this.status === 'ended') {
      throw new LongpipeError('control-failed', `setInputStream: pipeline is ${this.status}`)
    }
    this.inputStream = stream
    this.watchInputTrack()
    this.meter.reset()

    // Main-side raw passthrough (if active) follows the new camera.
    const mainRaw = this.status === 'recovering'
      || (this.status === 'passthrough' && this.activeBackend === 'none')
    if (mainRaw) this.output.startRaw(stream)

    if (this.pendingInput) {
      // The first worker hasn't consumed its input yet: rebuild it.
      this.pendingInput.cleanup()
      this.pendingInput = setupInput(this.topology.input, stream)
      return
    }
    // No GPU, raw from main: the worker reads no input (see onNoBackend).
    if (this.status === 'passthrough' && this.activeBackend === 'none' && !this.input) return
    const controller = this.controller
    if (!controller || controller.isTerminated || this.status === 'recovering') return   // next worker uses the new stream
    const next = setupInput(this.topology.input, stream)
    const gen = this.gen
    try {
      await controller.sendMessage('setInput', next.initFields, next.transferList)
    } catch (err) {
      next.cleanup()
      if (gen !== this.gen) return   // worker replaced meanwhile; it picks up this.inputStream
      throw err
    }
    if (gen !== this.gen) { next.cleanup(); return }
    this.input?.cleanup()
    this.input = next
  }

  // ── Control calls ─────────────────────────────────────────────────────────
  //
  // Every control call:
  //   - before `ready` / during recovery: QUEUES until the pipeline is
  //     operational (max TUNING.queueTimeoutMs), then runs;
  //   - with effects unavailable (no GPU, init failed, recovery gave up,
  //     camera ended, destroyed): rejects at once with that reason's type;
  //   - on failure: rejects with a LongpipeError ('control-failed' /
  //     'control-timeout' / 'background-load'), fires onError (non-fatal),
  //     and leaves the previous settings in place. `ready` is never affected.
  // Settings are remembered and re-applied after an automatic recovery.

  private async whenOperational(name: string): Promise<WorkerController> {
    for (;;) {
      switch (this.status) {
        case 'running':
          if (this.controller && !this.controller.isTerminated) return this.controller
          break
        case 'destroyed':
          throw new LongpipeError('control-failed', `${name}: pipeline destroyed`)
        case 'ended':
          throw new LongpipeError('camera-ended', `${name}: the input camera ended`)
        case 'passthrough': {
          const why = this.passthroughError
          throw new LongpipeError(why?.type ?? 'no-backend', `${name}: effects are unavailable — ${why?.message ?? 'no GPU backend'}`, { fatal: why?.fatal })
        }
      }
      let timer: ReturnType<typeof setTimeout> | null = null
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new LongpipeError('control-timeout',
          `${name}: the pipeline did not become ready within ${TUNING.queueTimeoutMs} ms`)), TUNING.queueTimeoutMs)
      })
      try {
        await Promise.race([this.op.promise, timeout])
      } finally {
        if (timer) clearTimeout(timer)
      }
    }
  }

  private control<T>(name: string, fn: (controller: WorkerController) => Promise<T>): Promise<T> {
    return quiet((async () => {
      try {
        const controller = await this.whenOperational(name)
        return await fn(controller)
      } catch (err) {
        throw this.reportControlError(name, err)
      }
    })())
  }

  // Normalize to a LongpipeError; report the control-call failure kinds via
  // onError (crash / passthrough reasons were already reported once).
  private reportControlError(name: string, err: unknown): LongpipeError {
    const e = isLongpipeError(err)
      ? err
      : new LongpipeError('control-failed', prefixMessage(`${name} failed`, errorMessage(err)), { cause: err })
    if (e.type === 'control-failed' || e.type === 'control-timeout' || e.type === 'background-load') {
      this.emitError(e)
    }
    return e
  }

  // A control call refused up front (wrong model, bad argument): typed, reported.
  private refuse(name: string, err: unknown): Promise<never> {
    return quiet(Promise.reject(this.reportControlError(name, err)))
  }

  private async normalizeOrFail(input: BackgroundInput, what: string): Promise<NormalizedBackground> {
    try {
      return await normalizeBackground(input, {
        onVideoError: (msg) => this.emitError(new LongpipeError('background-load', `${msg}; showing its last frame`)),
      })
    } catch (err) {
      throw new LongpipeError('background-load', `${what}: ${errorMessage(err)}; keeping the previous background`, { cause: err })
    }
  }

  // Background for a (re)starting worker. A failure falls back to 'none'
  // (the "previous" effect at startup) with a non-fatal error.
  private async normalizeDesiredBackground(): Promise<NormalizedBackground> {
    try {
      return await this.normalizeOrFail(this.desired.background, 'background')
    } catch (err) {
      const e = err as LongpipeError
      this.emitError(new LongpipeError('background-load',
        e.message.replace('keeping the previous background', "using 'none'"), { cause: e.cause }))
      this.desired.background = 'none'
      return { background: { kind: 'none' } }
    }
  }

  // Swap the background at runtime. Same wide input surface as construction;
  // resolves once the new background has been normalized (URL fetched, bitmap
  // decoded, video setup, etc.) and the worker has applied it. On failure the
  // previous background stays and the promise rejects ('background-load' /
  // 'control-failed' / 'control-timeout'). The previous background's
  // resources are cleaned up after the worker confirms the new one.
  setBackground(input: BackgroundInput): Promise<void> {
    return this.control('setBackground', async (controller) => {
      const norm = await this.normalizeOrFail(input, 'setBackground')
      const gen = this.gen
      const previousInput = this.desired.background
      this.desired.background = input
      try {
        await controller.sendMessage('setBackground', norm.background, norm.transferList ?? [])
      } catch (err) {
        norm.cleanup?.()
        if (gen === this.gen) this.desired.background = previousInput
        throw err
      }
      if (gen !== this.gen) { norm.cleanup?.(); return }
      const previousCleanup = this.bgCleanup
      this.bgCleanup = norm.cleanup ?? null
      previousCleanup?.()
    })
  }

  // Enable / update / disable the face touch-up at runtime. First call fetches
  // the landmark assets from weightsBaseUrl (cached for later param updates);
  // pass null to disable. Runs in parallel with the background effect. Requires
  // the loaded tier weights to carry a face blob (all current tiers do) — if
  // they don't, the worker silently leaves the effect off.
  setTouchup(opts: TouchupOptions | null): Promise<void> {
    if (opts !== null && isMediaPipe(this.model)) return this.refuse('setTouchup', requiresLongpipe('touchup', this.modelName))
    return this.control('setTouchup', async (controller) => {
      await this.sendTouchup(controller, opts)
      this.desired.touchup = opts
    })
  }

  private async sendTouchup(controller: WorkerController, opts: TouchupOptions | null): Promise<void> {
    if (opts === null) {
      await controller.sendMessage('setTouchup', null)
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
    // Busy: building the landmark nets compiles shaders synchronously.
    const assets = this.touchupAssets
    await this.busyCall(() => controller.sendMessage('setTouchup', { ...assets, params }))
  }

  // Enable / update / disable auto-reframe at runtime. `true` takes every
  // default; `false` (or null) turns it off. Needs no assets — it rides the face
  // heatmaps the tier already carries — so unlike setTouchup this never fetches.
  setReframe(cfg: ReframeConfig | null): Promise<void> {
    if (cfg && isMediaPipe(this.model)) return this.refuse('setReframe', requiresLongpipe('reframe', this.modelName))
    return this.control('setReframe', async (controller) => {
      await this.busyCall(() => controller.sendMessage('setReframe', normalizeReframe(cfg)))
      this.desired.reframe = cfg || null
    })
  }

  // Re-solve the frame now. This is the manual-mode entry point — the method to
  // wire to a "reframe" button. In auto mode the camera already tracks, so this
  // is a no-op there.
  reframe(): Promise<void> {
    if (isMediaPipe(this.model)) return this.refuse('reframe', requiresLongpipe('reframe', this.modelName))
    return this.control('reframe', (controller) => controller.sendMessage('reframeNow', undefined))
  }

  // Re-apply touch-up / reframe after a (re)start. Non-blocking: `ready` and
  // onRecovered never wait on touch-up assets.
  private replayFeatures(gen: number): void {
    const controller = this.controller
    if (!controller) return
    const report = (what: string) => (err: unknown) => {
      if (gen !== this.gen) return
      this.emitError(new LongpipeError('control-failed', `${what} init failed: ${errorMessage(err)}`, { cause: err }))
    }
    if (this.desired.reframe) {
      void this.busyCall(() => controller.sendMessage('setReframe', normalizeReframe(this.desired.reframe)))
        .catch(report('reframe'))
    }
    if (this.desired.touchup) {
      void this.sendTouchup(controller, this.desired.touchup).catch(report('touchup'))
    }
  }

  // Switch to a fixed preset. Fetches the preset's weights from
  // weightsBaseUrl unless you pass them. Stops the adaptive controller (an
  // explicit choice is respected from here on, including after recovery).
  // Rejects ('control-failed') on an unknown preset / weights failure; the
  // current preset keeps running. While disabled with no model loaded yet,
  // it only records the choice; setEnabled(true) loads it.
  setPreset(p: PresetName | ManualPreset, weights?: ArrayBuffer): Promise<void> {
    if (isMediaPipe(this.model)) return this.refuse('setPreset', requiresLongpipe('setPreset', this.modelName))
    this.adaptive?.stop()
    this.adaptive = null
    return this.control('setPreset', async (controller) => {
      if (p === 'auto') throw new LongpipeError('control-failed', "setPreset('auto') isn't supported at runtime; pass preset: 'auto' at construction")
      const manual = typeof p === 'string' ? resolveNamedPreset(p) : p
      if (!manual) throw new LongpipeError('control-failed', `setPreset: unknown preset '${String(p)}'`)
      const preset: ManualPreset = { ...manual, dtype: this.resolvedDtype }
      if (!this.modelLoaded && !this.modelLoading) {
        this.resolvedPreset = preset
        this.presetOverride = preset
        this.pendingWeights = weights ?? null
        return
      }
      const buf = weights ?? await fetchModelWeights(this.modelBaseUrl, preset.model, this.resolvedDtype)
      const res = await this.busyCall(() =>
        controller.sendMessage('setPreset', { preset, weights: buf }, [buf], { timeoutMs: TUNING.initTimeoutMs }))
      this.resolvedPreset = res.resolvedPreset
      this.presetOverride = res.resolvedPreset
    })
  }

  private startAdaptive(): void {
    this.adaptive?.stop()
    this.adaptive = null
    if (!this.adaptiveEligible || this.presetOverride || !this.resolvedPreset || !this.backendKind || !this.modelLoaded) return
    const dtype = this.resolvedDtype
    this.adaptive = new AdaptiveController({
      backendKind:  this.backendKind,
      initialModel: this.resolvedPreset.model,
      getInputFps:  () => this.meter.fps(),
      fetchWeights: (model) => fetchModelWeights(this.modelBaseUrl, model, dtype),
      getStats:     () => this.getStats(),
      swapPreset:   async (preset, weights) => {
        const controller = await this.whenOperational('adaptive')
        const res = await this.busyCall(() =>
          controller.sendMessage('setPreset', { preset, weights }, [weights], { timeoutMs: TUNING.initTimeoutMs }))
        this.resolvedPreset = res.resolvedPreset
      },
      onError: (err) => {
        // A swap cut short by a crash is part of that crash, not a new error.
        if (isLongpipeError(err.cause) && err.cause.type === 'worker-crash') return
        this.emitError(new LongpipeError('adaptive', err.message, { cause: err.cause }))
      },
    })
    this.adaptive.start()
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
  // A worker restart loses the canvas (it can't come back); onError says so.
  attachPreview(canvas: HTMLCanvasElement): Promise<void> {
    let offscreen: OffscreenCanvas
    try {
      offscreen = canvas.transferControlToOffscreen()
    } catch (err) {
      return this.refuse('attachPreview', err)
    }
    return this.control('attachPreview', async (controller) => {
      await controller.sendMessage('attachPreview', { canvas: offscreen }, [offscreen])
      this.previewAttached = true
    })
  }

  // Set / update the previewed candidate. `background` is the identical wide
  // surface as construction + setBackground (keyword, URL, ImageBitmap, color,
  // blur, image, video). 'none' previews the no-effect (raw) option. Resolves
  // once the worker has applied it (after any URL/bitmap load).
  setPreview(input: { background: BackgroundInput; fps?: number }): Promise<void> {
    return this.control('setPreview', async (controller) => {
      const norm = await this.normalizeOrFail(input.background, 'setPreview')
      try {
        await controller.sendMessage('setPreview', { background: norm.background, fps: input.fps }, norm.transferList ?? [])
      } catch (err) {
        norm.cleanup?.()
        throw err
      }
      const previousCleanup = this.previewBgCleanup
      this.previewBgCleanup = norm.cleanup ?? null
      previousCleanup?.()
    })
  }

  // Stop previewing. The preview canvas freezes on its last frame (hide it in
  // your UI). To "apply" a previewed effect, call setBackground(sameInput).
  clearPreview(): Promise<void> {
    return this.control('clearPreview', async (controller) => {
      await controller.sendMessage('clearPreview', {} as Record<string, never>)
      this.previewBgCleanup?.()
      this.previewBgCleanup = null
    })
  }

  // false = raw passthrough. The first setEnabled(true) on a pipeline started
  // with enabled: false fetches the weights first.
  setEnabled(on: boolean): Promise<void> {
    return this.control('setEnabled', async (controller) => {
      if (on && !this.modelLoaded) {
        const gen = this.gen
        this.modelLoading ??= this.loadModel(controller, gen).finally(() => { this.modelLoading = null })
        await this.modelLoading
        if (gen === this.gen) this.startAdaptive()   // else the next send rejects (worker gone)
      }
      await controller.sendMessage('setEnabled', { enabled: on })
      this.desired.enabled = on
    })
  }

  // Stats snapshot: the worker's rolling counters (fps, model time, preset…)
  // plus main-side state (status, backend, measured input fps, restarts).
  // Never rejects; while the worker isn't rendering (recovery, no GPU) the
  // numbers describe the raw passthrough.
  async getStats(): Promise<PipelineStats> {
    const inputFps = this.meter.fps()
    const main = {
      status:   this.status,
      inputFps,
      restarts: this.restarts,
      backend:  (this.status === 'running' ? this.activeBackend : 'none') ?? 'none',
    }
    const controller = this.controller
    let worker: RendererStats | null = null
    if (controller && !controller.isTerminated && this.status !== 'recovering') {
      try {
        worker = await controller.sendMessage('getStats', {} as Record<string, never>, [], { timeoutMs: 2000 })
      } catch { worker = null }
    }
    // While the main side serves raw frames (recovering, or no GPU on a
    // relay / main-canvas path) the output rate is the input rate.
    const mainRaw = this.status === 'recovering'
      || (this.status === 'passthrough' && (!worker || (worker.backend === 'none' && worker.fps === 0)))
    return {
      fps:        mainRaw || !worker ? Math.round(inputFps ?? 0) : worker.fps,
      modelFps:   worker?.modelFps ?? 0,
      modelMs:    worker?.modelMs ?? 0,
      displayMs:  worker?.displayMs ?? 0,
      skipped:    worker?.skipped ?? 0,
      preset:     worker?.preset ?? 'none',
      skipFrames: worker?.skipFrames ?? 0,
      latencyMs:  mainRaw ? 0 : worker?.latencyMs ?? 0,
      enabled:    worker?.enabled ?? false,
      inputPath:  this.topology.input,
      outputPath: this.output.path,
      ...main,
    }
  }

  private readyInfo(): ReadyInfo {
    return {
      backend: this.activeBackend ?? 'none',
      dtype:   this.activeBackend && this.activeBackend !== 'none' ? this.resolvedDtype : null,
      preset:  this.activeBackend === 'none' || !this.modelLoaded ? null
             : isMediaPipe(this.model) ? `mediapipe-selfie-${this.model.variant}`
             : this.resolvedPreset?.model ?? null,
    }
  }

  private emitError(e: LongpipeError): void {
    if (e.fatal) console.error(`[longpipe/pipeline] ${e.type}:`, e.message)
    else console.warn(`[longpipe/pipeline] ${e.type}:`, e.message)
    safeCall(() => this.opts.onError?.(e))
  }

  // ── Audio ─────────────────────────────────────────────────────────────────

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

  // Resume the denoise AudioContext if the autoplay policy suspended it (the
  // output carries the original audio meanwhile; see onAudioTrackChanged).
  // Call from a user gesture (join / unmute button). The SDK also resumes on
  // the first gesture by itself. No-op without audio: 'denoise'.
  async resumeAudio(): Promise<void> {
    try {
      await this.denoiser?.resume()
    } catch (err) {
      const e = new LongpipeError('audio', prefixMessage('resumeAudio failed', errorMessage(err)), { cause: err })
      this.emitError(e)
      throw e
    }
  }

  // Latest per-hop audio telemetry (p50/p95 ms vs the render-quantum deadline,
  // buffered latency, resolved sample rate). null until the worklet reports, or
  // when denoise isn't active. Synchronous — the worklet pushes stats; we cache.
  getAudioStats(): AudioStats | null {
    return this.denoiser?.getStats() ?? null
  }

  // ── Test hooks ────────────────────────────────────────────────────────────

  // Simulate a failure inside the worker (tests only; not part of the API).
  /** @internal */
  _debug(action: 'lose-backend' | 'hang' | 'throw' | 'terminate'): Promise<void> {
    if (action === 'terminate') {
      // Kill the worker the way an OOM kill would: no error event, no reply.
      this.worker?.terminate()
      return Promise.resolve()
    }
    return this.controller
      ? this.controller.sendMessage('debug', { action }).then(() => {})
      : Promise.reject(new LongpipeError('control-failed', '_debug: no worker'))
  }

  destroy(): void {
    if (this.status === 'destroyed') return
    this.status = 'destroyed'
    this.gen++
    if (this.watchdog) clearInterval(this.watchdog)
    this.watchdog = null
    if (this.restartTimer) clearTimeout(this.restartTimer)
    this.restartTimer = null
    this.unwatchInput?.()
    this.adaptive?.stop()
    this.adaptive = null
    this.denoiser?.destroy()
    this.denoiser = null
    this.bgCleanup?.()
    this.bgCleanup = null
    this.previewBgCleanup?.()
    this.previewBgCleanup = null
    // The worker gets a 'destroy' (releases its frames + GPU) before it's killed.
    this.controller?.terminate('pipeline destroyed', TUNING.retireGraceMs)
    this.controller = null
    this.input?.cleanup()
    this.pendingInput?.cleanup()
    this.output.cleanup()
    for (const o of this.retiredOutputs) o.cleanup()
    this.op.reject(new LongpipeError('control-failed', 'pipeline destroyed'))
  }
}

function normalizeReframe(cfg: ReframeConfig | null) {
  return cfg === true ? {} : (cfg === false || cfg === null ? null : cfg)
}

function safeCall(fn: () => void): void {
  try { fn() } catch (err) { console.error('[longpipe/pipeline] callback threw:', err) }
}

// Public type re-exports
export type { PipelineOptions as Options }
export type {
  BackgroundInput, Background, BlurInput, ColorInput, ImageInput, VideoInput,
}                                                             from './background'
export type { PresetName, ManualPreset, ModelName }           from './presets'
export type { ModelChoice }                                   from './model_choice'
export type { AudioMode, AudioInput, DenoiseOptions }         from './audio'
export type { DenoiseModel, DenoiseTier, DenoiseModelOption } from '~/audio/kernels.ts'
export type { AudioStats }                                    from '~/audio/messages.ts'
export type { PipelineError, ErrorSource, PipelineErrorType } from './errors'
export { LongpipeError }                                      from './errors'
export type { ActiveBackend, PipelineStats, PipelineStatus, ReadyInfo } from './messages'
