import { tagFrameOrientation, type Orientation } from '~/model/orientation.ts'
import { tagFrameEntry } from '../latency'

// Per-frame postMessage input adapter.
//
// Main thread reads frames from a hidden <video> via requestVideoFrameCallback
// and posts them to a dedicated MessagePort. Worker wraps that port in a
// synthetic ReadableStream<VideoFrame>.
//
// highWaterMark: 0 + replace-on-arrival → late frames drop at the source,
// matching MSTP's behavior for videoconf-correct latency.

export function createPostMessageInput(port: MessagePort): ReadableStream<VideoFrame> {
  let pending: VideoFrame | null = null
  let resolvePull: ((frame: VideoFrame) => void) | null = null
  // Set once the consumer cancels. Frames already in flight on the port can
  // still arrive after that (seen in Firefox during GPU-loss recovery); close
  // them instead of enqueueing into a closed stream.
  let closed = false

  port.onmessage = (e: MessageEvent<{ frame: VideoFrame; orientation?: Orientation; sentAt?: number }>) => {
    const { frame, orientation, sentAt } = e.data
    if (closed) { frame?.close(); return }
    // When main created the frame: the start of getStats().latencyMs.
    tagFrameEntry(frame, sentAt)
    // Main measured the display orientation the frame still needs (Safari's
    // hidden camera rotation); the renderer looks it up per frame.
    tagFrameOrientation(frame, orientation)
    if (resolvePull) {
      const r = resolvePull
      resolvePull = null
      r(frame)
    } else {
      // Replace-on-arrival: drop the stale pending frame, hold the new one.
      if (pending) pending.close()
      pending = frame
    }
  }
  port.start?.()

  return new ReadableStream<VideoFrame>({
    pull(controller) {
      if (pending) {
        controller.enqueue(pending)
        pending = null
        return
      }
      return new Promise<void>((resolve) => {
        resolvePull = (frame) => {
          if (closed) { frame.close(); resolve(); return }
          try {
            controller.enqueue(frame)
          } catch {
            // Stream closed/errored underneath us: drop the frame quietly.
            closed = true
            frame.close()
          }
          resolve()
        }
      })
    },
    cancel() {
      closed = true
      resolvePull = null
      if (pending) { pending.close(); pending = null }
      port.close()
    },
  }, { highWaterMark: 0 })
}
