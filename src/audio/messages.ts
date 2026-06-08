// AudioWorklet port protocol. The denoise worklet is driven over its node's
// MessagePort (not the video WorkerController) — it's a tiny, fire-and-forget
// surface, so a plain discriminated union beats the UUID req/res machinery.

import type { DenoiseModel } from './kernels'

// Main thread → processor.
export type ToWorklet =
  | { type: 'config'; postFilterBeta?: number; gruLeak?: number }
  | { type: 'enabled'; value: boolean }   // false = dry passthrough (cheap A/B / bypass)

// Processor → main thread.
export type FromWorklet =
  | { type: 'ready' }
  | { type: 'stats'; stats: AudioStats }
  | { type: 'error'; message: string }

// Per-hop telemetry, surfaced via EffectsPipeline.getAudioStats().
export interface AudioStats {
  model:       DenoiseModel
  // Per-hop compute time vs the render-quantum deadline (~2.67 ms). Available
  // only where the worklet scope exposes performance.now() (recent Chrome).
  p50Ms:       number | null
  p95Ms:       number | null
  // End-to-end buffered latency added by the 128⇄480 ring (ms).
  latencyMs:   number
  // Whether the kernel is actively processing (false during passthrough/bypass).
  active:      boolean
  // Resolved AudioContext sample rate; ≠ 48000 means the resampler is engaged.
  sampleRate:  number
}

// Constructed once on the main thread, passed into the processor via
// processorOptions (structured-cloned). The wasm Module + weights are compiled/
// fetched on the main thread because the worklet scope can't fetch.
export interface ProcessorInit {
  model:           DenoiseModel
  module:          WebAssembly.Module
  weights:         ArrayBuffer | null
  enabled:         boolean
  postFilterBeta?: number
  gruLeak?:        number
}
