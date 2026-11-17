// Per-frame pipeline latency: input frame in → its output frame handed to the
// output transport, measured in the worker (getStats().latencyMs).
//
// Start of the clock, per input path:
//   'rvfc-postmessage' — when main created the VideoFrame in its rVFC callback
//                        (main sends that time, as an absolute timeOrigin-based
//                        time, with the frame; tagged on arrival). Includes the
//                        postMessage hop and any wait in the replace-on-arrival slot.
//   'mstp'             — when the output sink starts on the frame (the pipe pulls
//                        one frame at a time, so frames don't queue on our side;
//                        MSTP's own buffer before that isn't visible to us).
// End of the clock: the frame is submitted to the output (VideoFrame posted to
// the relay, bitmap posted to main, or drawn into the captured canvas). GPU
// completion and the browser's capture/compositor step come after that.
//
// Cost: one WeakMap entry per postMessage frame, one subtraction per output
// frame, and a ~1 s window of samples (≤ fps entries).

const entryTime = new WeakMap<object, number>()

// rvfc-postmessage: `sentAt` is main's performance.timeOrigin + now().
export function tagFrameEntry(frame: object, sentAt: number | undefined): void {
  if (typeof sentAt === 'number') entryTime.set(frame, sentAt - performance.timeOrigin)
}

// When the frame entered the pipeline (this worker's performance.now() clock);
// `fallback` (usually now()) when it wasn't tagged.
export function frameEntryTime(frame: object, fallback: number): number {
  return entryTime.get(frame) ?? fallback
}

const WINDOW_MS = 1000
const samples: { t: number; ms: number }[] = []

export function recordLatency(entry: number | undefined): void {
  if (entry === undefined) return
  const now = performance.now()
  const ms = now - entry
  // Clocks of two contexts can disagree by a hair; ignore nonsense.
  if (!(ms >= 0 && ms < 10_000)) return
  samples.push({ t: now, ms })
  trim(now)
}

// Rolling mean over the last second, in ms (0 when no frames).
export function meanLatencyMs(): number {
  trim(performance.now())
  if (samples.length === 0) return 0
  let sum = 0
  for (const s of samples) sum += s.ms
  return Math.round((sum / samples.length) * 10) / 10
}

function trim(now: number): void {
  while (samples.length && now - samples[0].t > WINDOW_MS) samples.shift()
}
