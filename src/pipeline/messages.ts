// Message taxonomy for the main↔worker control plane.
//
// Data plane (frames, bitmaps) flows through transferred streams or
// dedicated MessagePorts and is NOT modeled here — that bypass is
// intentional, to avoid UUID/Promise overhead per frame.

import type { Dtype } from '~/model/backend'
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
}

// ── Persistent events from worker → main ────────────────────────────────────

export interface EventMap {
  ready: void
  stats: RendererStats
  error: { message: string; recoverable: boolean }
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
