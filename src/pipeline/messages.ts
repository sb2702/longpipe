import type { ReframeConfig } from './worker/renderer'
export type { ReframeConfig }
// Message taxonomy for the main↔worker control plane.
//
// Data plane (frames, bitmaps) flows through transferred streams or
// dedicated MessagePorts and is NOT modeled here — that bypass is
// intentional, to avoid UUID/Promise overhead per frame.

import type { Dtype, FaceTouchupParams } from '~/model/backend.ts'
import type { ManualPreset, PresetName } from './presets'
import type { Background } from './background'
import type { Topology } from './topology'
import type { ResolvedModel } from './model_choice'
import type { PipelineError, PipelineErrorType } from './errors'
export type { PipelineError, ErrorSource, PipelineErrorType } from './errors'

// ── Command-keyed maps ──────────────────────────────────────────────────────
// Type-safe sendMessage<C>(cmd, data) → Promise<response>.

export interface CmdDataMap {
  init:          InitData
  startRender:   { weights: ArrayBuffer; preset?: ManualPreset }   // preset: overrides init's (setPreset while disabled)
  setBackground: Background
  setEnabled:    { enabled: boolean }
  setPreset:     { preset: PresetName | ManualPreset; weights?: ArrayBuffer }
  getStats:      Record<string, never>
  destroy:       Record<string, never>
  // Preview: attach the (transferred) preview canvas once, then set/clear the
  // candidate effect. setPreview carries a canonical Background (same shape as
  // setBackground) plus an optional throttle fps.
  attachPreview: { canvas: OffscreenCanvas }
  setPreview:    { background: Background; fps?: number }
  clearPreview:  Record<string, never>
  // Face touch-up: full payload enables/updates (assets are cached main-side
  // and structured-cloned — no transfer, so param updates can resend them);
  // null disables. The worker assembles a FaceEffectsConfig for the renderer.
  setTouchup:    TouchupPayload | null
  setReframe:    ReframeConfig | null
  reframeNow:    void
  // Swap the input frame source (camera switch) without touching the output.
  setInput:      { inputReadable?: ReadableStream<VideoFrame>; inputPort?: MessagePort }
  // Test hooks (simulate failures). Not used by the SDK itself.
  debug:         { action: 'lose-backend' | 'hang' | 'throw' }
}

export interface CmdResponseMap {
  init:          InitResponse
  startRender:   void
  setBackground: void
  setEnabled:    void
  setPreset:     PresetSwapResult
  getStats:      RendererStats
  destroy:       void
  attachPreview: void
  setPreview:    void
  clearPreview:  void
  setTouchup:    void
  setReframe:    void
  reframeNow:    void
  setInput:      void
  debug:         void
}

// Touch-up wire payload — the flattened FaceTopology (typed arrays and
// ImageBitmap structured-clone across postMessage) + landmark weights + params.
export interface TouchupPayload {
  landmarkWeights: ArrayBuffer
  topoCount:       number
  topoUv:          Float32Array
  topoIdx:         Float32Array
  weightMask:      ImageBitmap
  params:          FaceTouchupParams
}

export type CmdName = keyof CmdDataMap

// ── Init payload — transport endpoints + initial config ─────────────────────

export interface InitData {
  topology:   Topology
  // Segmentation model. MediaPipe skips autotune (resolvedPreset → null) and
  // startRender's weights are a MediaPipe pack.
  model:      ResolvedModel
  preset:     PresetName | ManualPreset
  background: Background
  enabled:    boolean
  // Output canvas dimensions. For transfer-capture topology this is
  // redundant with outputCanvas (the transferred OffscreenCanvas
  // already has its size); for MSTG and bitmap-shuttle the worker
  // allocates its own OffscreenCanvas and uses these dimensions.
  canvasSize: { w: number; h: number }

  // Backend + dtype are *preferences*. Worker's setup_backend honors them
  // when possible and falls back when not. 'auto' = pick the best available
  // (try WebGPU → fall back to WebGL). Resolved values come back via
  // InitResponse so main can log/display what's actually running.
  // 'none' skips GPU setup entirely (test hook for the no-GPU path).
  backend:  'webgpu' | 'webgl' | 'auto' | 'none'
  dtype:    Dtype

  // When true, the worker enables the same debug-log gate as the main thread
  // (both contexts have their own setDebug state). Default false.
  debug?:   boolean

  // Heartbeat cadence for main's watchdog (ms). Default 1000.
  heartbeatMs?: number

  // Note: weights are NOT in InitData. Two-phase init: 'init' resolves
  // backend + preset and returns InitResponse; main then fetches weights
  // for the resolved preset and sends them via 'startRender'. This lets
  // autotune (when implemented) pick the preset before main commits to a
  // weights URL.

  // Transport endpoints — presence depends on topology. Transferred via
  // the postMessage transfer list at init time.
  inputReadable?:  ReadableStream<VideoFrame>      // topology.input  === 'mstp'
  inputPort?:      MessagePort                     // topology.input  === 'rvfc-postmessage'
  outputFramePort?: MessagePort                    // topology.output === 'mstg' (VideoFrames → relay worker)
  outputCanvas?:   OffscreenCanvas                 // topology.output === 'transfer-capture'
  outputPort?:     MessagePort                     // topology.output === 'bitmap-shuttle'
}

export type BackendKind = 'webgpu' | 'webgl'
// Public backend name ('webgl2', not the internal 'webgl').
export type ActiveBackend = 'webgpu' | 'webgl2' | 'none'

export interface InitResponse {
  resolvedPreset:  ManualPreset | null             // 'auto' resolves to a concrete preset; null for MediaPipe models / no backend
  resolvedBackend: BackendKind | 'none'            // what the worker actually got ('none' = no usable GPU)
  resolvedDtype:   Dtype                           // patched if downgraded (f16 → f32)
  // 'none' backend only: whether the worker draws raw frames to its output
  // itself (transfer-capture with a 2D context). When false, the main thread
  // provides the raw passthrough.
  workerDrawsRaw?: boolean
  // 'none' backend only: why GPU setup failed.
  backendError?:   string
}

export interface PresetSwapResult {
  resolvedPreset: ManualPreset
}

export interface RendererStats {
  // Rolling 1-second window
  fps:           number    // process() calls per second (≈ source FPS)
  modelFps:      number    // model runs per second
  modelMs:       number    // median model run time (ms)
  displayMs:     number    // median runDisplay time (ms)
  skipped:       number    // total skipped-model frames since init

  // Static state (snapshot at last preset swap)
  preset:        string    // resolved preset's model name
  skipFrames:    number    // resolved preset's skipFrames
  enabled:       boolean   // false = passthrough mode

  // Transport identifiers — useful for debugging which paths the topology
  // selector picked on a given browser. Static across the session.
  inputPath:     string    // 'mstp' | 'rvfc-postmessage'
  outputPath:    string    // 'mstg' | 'transfer-capture' | 'bitmap-shuttle'
  backend:       ActiveBackend

  // Pipeline latency (ms, rolling mean over ~1 s): an input frame entering the
  // pipeline → its output frame submitted to the output transport. Starts when
  // main grabs the frame (rvfc-postmessage) or when the worker takes it from
  // MSTP (mstp); ends before GPU completion / the browser's capture step. See
  // worker/latency.ts. 0 when no frames are flowing through the worker (e.g.
  // while main serves raw passthrough).
  latencyMs:     number
}

export type PipelineStatus =
  | 'starting'     // constructed; worker booting (output: raw / unprocessed)
  | 'running'      // effect pipeline live
  | 'passthrough'  // effects unavailable (no GPU, init failed, recovery gave up); output: raw camera
  | 'recovering'   // worker restarting / backend being recreated; output: raw camera
  | 'ended'        // input camera ended; output ended
  | 'destroyed'

// What getStats() returns: the worker's renderer counters plus main-side state.
export interface PipelineStats extends RendererStats {
  status:    PipelineStatus
  // Measured input (camera) frame rate, or null if it can't be measured here.
  inputFps:  number | null
  // Automatic restarts so far (worker crash / backend loss).
  restarts:  number
}

export interface ReadyInfo {
  backend: ActiveBackend
  dtype:   Dtype | null
  preset:  string | null     // resolved model name ('small', …, 'mediapipe-selfie-…'), null with no backend
}

// ── Persistent events from worker → main ────────────────────────────────────

export interface EventMap {
  ready:      void
  stats:      RendererStats
  error:      WorkerErrorEvent
  // Liveness + progress, every TUNING.heartbeatMs. framesOut counts frames
  // the worker wrote to the output (effect or raw).
  heartbeat:  { framesOut: number }
  // The input stream closed (camera ended / stopped).
  inputEnded: void
}

// Error event from the worker. `type` drives main's reaction: 'backend-lost'
// and 'worker-crash' (pipe failure) trigger recovery; others are reported.
export interface WorkerErrorEvent {
  type:    PipelineErrorType
  message: string
  source?: PipelineError['source']
  fatal?:  boolean
  cause?:  unknown
}

export type EventName = keyof EventMap

// ── Wire types — what actually crosses postMessage ──────────────────────────

export interface WorkerRequest<C extends CmdName = CmdName> {
  cmd:        C
  data:       CmdDataMap[C]
  request_id: string
}

export interface WorkerResponse<C extends CmdName = CmdName> {
  request_id: string
  res?:       CmdResponseMap[C]
  // Present when the command threw: the call's promise rejects with it.
  error?:     { message: string }
}

export interface WorkerEvent<E extends EventName = EventName> {
  request_id: E       // persistent listener key matches event name
  res:        EventMap[E]
}
