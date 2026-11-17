// Internal timing knobs for the robustness layer (timeouts, watchdog, restart
// backoff) plus test-only overrides. NOT part of the public API (not exported
// from the package entry); tests import this module directly and mutate it.

import type { Topology } from './topology'
import type { Orientation } from '~/model/orientation.ts'

export const TUNING = {
  // Control calls (setBackground, setPreset, …): max wait for the worker's
  // answer once the call has been sent.
  controlTimeoutMs: 10_000,
  // Max time a control call made before `ready` (or during recovery) waits in
  // the queue for the pipeline to become operational.
  queueTimeoutMs:   30_000,
  // init / startRender round trips (autotune + shader compile can be slow).
  initTimeoutMs:    30_000,
  // Worker heartbeat cadence, and how long main waits without one before
  // treating the worker as hung. Only armed while running and no heavy
  // command (startRender / setPreset) is in flight.
  heartbeatMs:      1_000,
  watchdogMs:       5_000,
  // A worker being replaced or destroyed gets this long to release its
  // frames (and answer 'destroy') before it is terminated.
  retireGraceMs:    500,
  // Frames stalled: input advanced while worker output didn't, for this long.
  stallMs:          3_000,
  // Automatic restarts (worker crash / backend lost) before giving up, and the
  // delay before each attempt. The attempt counter resets after
  // `stableResetMs` of healthy running.
  maxRestarts:      3,
  restartBackoffMs: [250, 1_000, 3_000] as number[],
  stableResetMs:    60_000,
  // Test-only: force a transport pair or a backend ('none' simulates no GPU).
  forceTopology:    null as Topology | null,
  forceBackend:     null as 'webgpu' | 'webgl' | 'none' | null,
  // Test-only: on the rvfc-postmessage input, tag every frame with this hidden
  // orientation instead of measuring it (simulates Safari's camera frames).
  simulateHiddenOrientation: null as Orientation | null,
}

export const TUNING_DEFAULTS = { ...TUNING, restartBackoffMs: [...TUNING.restartBackoffMs] }

export function resetTuning(): void {
  Object.assign(TUNING, { ...TUNING_DEFAULTS, restartBackoffMs: [...TUNING_DEFAULTS.restartBackoffMs] })
}
