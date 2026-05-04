// Message taxonomy for the main↔worker control plane.
//
// Data plane (frames, bitmaps) flows through transferred streams or
// dedicated MessagePorts and is NOT modeled here — that bypass is
// intentional, to avoid UUID/Promise overhead per frame.

import type { Dtype } from '~/model/backend'
import type { ManualPreset, PresetName } from './presets'
import type { EffectConfig } from './effects'
import type { Topology } from './topology'

// ── Command-keyed maps ──────────────────────────────────────────────────────
// Type-safe sendMessage<C>(cmd, data) → Promise<response>.

export interface CmdDataMap {
  init:       InitData
  setEffect:  EffectConfig
  setEnabled: { enabled: boolean }
  setPreset:  { preset: PresetName | ManualPreset; weights?: ArrayBuffer }
  getStats:   Record<string, never>
  destroy:    Record<string, never>
}

export interface CmdResponseMap {
  init:       InitResponse
  setEffect:  void
  setEnabled: void
  setPreset:  PresetSwapResult
  getStats:   RendererStats
  destroy:    void
}

export type CmdName = keyof CmdDataMap

// ── Init payload — transport endpoints + initial config ─────────────────────

export interface InitData {
  topology: Topology
  preset:   PresetName | ManualPreset
  effect:   EffectConfig
  enabled:  boolean
  weights:  ArrayBuffer | null   // initial preset's weights (null when preset === 'auto')
  backend:  'webgpu' | 'webgl'
  dtype:    Dtype

  // Transport endpoints — presence depends on topology. Transferred via
  // the postMessage transfer list at init time.
  inputReadable?:  ReadableStream<VideoFrame>      // topology.input  === 'mstp'
  inputPort?:      MessagePort                     // topology.input  === 'rvfc-postmessage'
  outputWritable?: WritableStream<VideoFrame>      // topology.output === 'mstg'
  outputCanvas?:   OffscreenCanvas                 // topology.output === 'transfer-capture'
  outputPort?:     MessagePort                     // topology.output === 'bitmap-shuttle'
}

export interface InitResponse {
  resolvedPreset: ManualPreset       // 'auto' resolves to a concrete preset
}

export interface PresetSwapResult {
  resolvedPreset: ManualPreset
}

export interface RendererStats {
  fps:     number
  modelMs: number
  skipped: number
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
