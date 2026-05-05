// Message taxonomy for the main↔worker control plane.
//
// Data plane (frames, bitmaps) flows through transferred streams or
// dedicated MessagePorts and is NOT modeled here — that bypass is
// intentional, to avoid UUID/Promise overhead per frame.

import type { Dtype } from '~/model/backend.ts'
import type { ManualPreset, PresetName } from './presets'
import type { Background } from './background'
import type { Topology } from './topology'

// ── Command-keyed maps ──────────────────────────────────────────────────────
// Type-safe sendMessage<C>(cmd, data) → Promise<response>.

export interface CmdDataMap {
  init:          InitData
  startRender:   { weights: ArrayBuffer }
  setBackground: Background
  setEnabled:    { enabled: boolean }
  setPreset:     { preset: PresetName | ManualPreset; weights?: ArrayBuffer }
  getStats:      Record<string, never>
  destroy:       Record<string, never>
}

export interface CmdResponseMap {
  init:          InitResponse
  startRender:   void
  setBackground: void
  setEnabled:    void
  setPreset:     PresetSwapResult
  getStats:      RendererStats
  destroy:       void
}

export type CmdName = keyof CmdDataMap

// ── Init payload — transport endpoints + initial config ─────────────────────

export interface InitData {
  topology:   Topology
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
  backend:  'webgpu' | 'webgl' | 'auto'
  dtype:    Dtype

  // Note: weights are NOT in InitData. Two-phase init: 'init' resolves
  // backend + preset and returns InitResponse; main then fetches weights
  // for the resolved preset and sends them via 'startRender'. This lets
  // autotune (when implemented) pick the preset before main commits to a
  // weights URL.

  // Transport endpoints — presence depends on topology. Transferred via
  // the postMessage transfer list at init time.
  inputReadable?:  ReadableStream<VideoFrame>      // topology.input  === 'mstp'
  inputPort?:      MessagePort                     // topology.input  === 'rvfc-postmessage'
  outputWritable?: WritableStream<VideoFrame>      // topology.output === 'mstg'
  outputCanvas?:   OffscreenCanvas                 // topology.output === 'transfer-capture'
  outputPort?:     MessagePort                     // topology.output === 'bitmap-shuttle'
}

export interface InitResponse {
  resolvedPreset:  ManualPreset                    // 'auto' resolves to a concrete preset
  resolvedBackend: 'webgpu' | 'webgl'              // what the worker actually got
  resolvedDtype:   Dtype                           // patched if downgraded (f16 → f32)
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
}

// ── Public error type — surfaced to callers via PipelineOptions.onError ─────

// `source` categorizes where the error originated so callers can branch
// (e.g. log differently, surface UI, retry only certain kinds). 'unknown'
// is the catchall for errors that don't fit a narrower bucket.
export type ErrorSource =
  | 'backend-lost'   // WebGL context lost / WebGPU device lost
  | 'worker'         // generic worker / pipe failure
  | 'adaptive'       // adaptive preset swap failed
  | 'background'    // background setup or runtime issue (e.g. video track ended)
  | 'unknown'

export interface PipelineError {
  message:     string
  source:      ErrorSource
  // false = the pipeline cannot continue; caller should destroy() + recreate.
  // true  = transient; pipeline keeps running in degraded state.
  recoverable: boolean
  // Original Error / DOMException / etc. — opaque, for debugging.
  cause?:      unknown
}

// ── Persistent events from worker → main ────────────────────────────────────

export interface EventMap {
  ready: void
  stats: RendererStats
  error: PipelineError
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
  res:        CmdResponseMap[C]
}

export interface WorkerEvent<E extends EventName = EventName> {
  request_id: E       // persistent listener key matches event name
  res:        EventMap[E]
}
