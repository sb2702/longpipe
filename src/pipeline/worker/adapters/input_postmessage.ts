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

  port.onmessage = (e: MessageEvent<{ frame: VideoFrame }>) => {
    const { frame } = e.data
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
          controller.enqueue(frame)
          resolve()
        }
      })
    },
    cancel() {
      if (pending) { pending.close(); pending = null }
      port.close()
    },
  }, { highWaterMark: 0 })
}
