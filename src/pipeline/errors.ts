// Typed errors. Every failure the pipeline can survive or report goes through
// one shape — PipelineError — delivered to `onError`, and (for control calls)
// as the rejection value of that call's promise.
//
// Policy: FAIL OPEN. The output track always carries something: the effect when
// possible, otherwise the raw camera frames / original audio. `fatal` says
// whether the pipeline will get the effect back on its own:
//   fatal: false → the SDK has degraded (raw passthrough, previous settings
//                  kept, …) and is handling it — log it, maybe show a hint.
//   fatal: true  → effects won't come back without a new pipeline (or, for
//                  'camera-ended', the output has ended). Output video (if any)
//                  is still the raw camera.

// What happened. Stable strings — branch on these, not on `message`.
export type PipelineErrorType =
  | 'no-backend'       // no usable GPU backend (WebGPU / WebGL2) at startup → raw passthrough
  | 'init-failed'      // weights download / autotune / worker init failed → raw passthrough
  | 'background-load'  // background image/video failed to load or decode → previous kept
  | 'backend-lost'     // GPU device / WebGL context lost → raw passthrough, recovering
  | 'worker-crash'     // worker errored, stalled or stopped responding → raw passthrough, restarting
  | 'recovery-failed'  // automatic recovery gave up → raw passthrough stays
  | 'camera-ended'     // the input video track ended → output video track ended
  | 'control-failed'   // a setX() call failed → previous settings kept
  | 'control-timeout'  // a setX() call got no answer in time → previous settings kept
  | 'adaptive'         // adaptive preset swap failed → stays on the current preset
  | 'audio'            // audio denoise failed → original audio passes through
  | 'audio-suspended'  // AudioContext suspended by autoplay policy → original audio (new track) until resumeAudio() / a gesture

// Where it happened (coarse; kept from 0.0.18 for compatibility).
export type ErrorSource =
  | 'backend-lost'   // WebGL context lost / WebGPU device lost
  | 'worker'         // worker / pipe / init failure
  | 'adaptive'       // adaptive preset swap
  | 'background'     // background load / decode
  | 'audio'          // audio denoise subsystem
  | 'camera'         // the input stream
  | 'control'        // a control call (setPreset, setTouchup, …)
  | 'unknown'

export interface PipelineError {
  type:        PipelineErrorType
  // true = effects won't return without recreating the pipeline (see above).
  fatal:       boolean
  message:     string
  source:      ErrorSource
  // Deprecated alias of !fatal (0.0.18 name). Kept so existing handlers work.
  recoverable: boolean
  // Original Error / DOMException / etc. Opaque, for debugging.
  cause?:      unknown
}

const DEFAULT_SOURCE: Record<PipelineErrorType, ErrorSource> = {
  'no-backend':      'worker',
  'init-failed':     'worker',
  'background-load': 'background',
  'backend-lost':    'backend-lost',
  'worker-crash':    'worker',
  'recovery-failed': 'worker',
  'camera-ended':    'camera',
  'control-failed':  'control',
  'control-timeout': 'control',
  'adaptive':        'adaptive',
  'audio':           'audio',
  'audio-suspended': 'audio',
}

const DEFAULT_FATAL: Record<PipelineErrorType, boolean> = {
  'no-backend':      false,
  'init-failed':     true,
  'background-load': false,
  'backend-lost':    false,
  'worker-crash':    false,
  'recovery-failed': true,
  'camera-ended':    true,
  'control-failed':  false,
  'control-timeout': false,
  'adaptive':        false,
  'audio':           false,
  'audio-suspended': false,
}

// An Error subclass that also satisfies PipelineError, so a rejected control
// call can be `catch`-ed as an Error AND branched on by `type`.
export class LongpipeError extends Error implements PipelineError {
  readonly type:        PipelineErrorType
  readonly fatal:       boolean
  readonly source:      ErrorSource
  readonly recoverable: boolean
  readonly cause?:      unknown

  constructor(type: PipelineErrorType, message: string, opts: { fatal?: boolean; source?: ErrorSource; cause?: unknown } = {}) {
    super(message)
    this.name        = 'LongpipeError'
    this.type        = type
    this.fatal       = opts.fatal ?? DEFAULT_FATAL[type]
    this.source      = opts.source ?? DEFAULT_SOURCE[type]
    this.recoverable = !this.fatal
    this.cause       = opts.cause
  }
}

export function isLongpipeError(e: unknown): e is LongpipeError {
  return e instanceof LongpipeError
}

export function errorMessage(e: unknown): string {
  return (e as Error)?.message ?? String(e)
}

// `${prefix}: ${message}`, unless the message already starts with that prefix
// (the worker prefixes its own failures, e.g. "init failed: …").
export function prefixMessage(prefix: string, message: string): string {
  return message.startsWith(`${prefix}:`) ? message : `${prefix}: ${message}`
}
