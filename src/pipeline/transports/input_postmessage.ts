// Main-side setup for the rVFC + postMessage input transport.
//
// Pairs with worker/adapters/input_postmessage.ts. Universal input fallback
// (used when MediaStreamTrackProcessor isn't available — i.e. Firefox /
// Safari).
//
// What main thread owns:
//   - A hidden HTMLVideoElement playing the input MediaStream
//   - requestVideoFrameCallback loop that creates a fresh VideoFrame per
//     new video frame and ships it (transferred, zero-copy) to the worker
//     via a MessagePort
//
// Backpressure: postMessage is fire-and-forget. The worker-side adapter
// has a "replace pending on arrival" slot so frames don't queue
// indefinitely on its side, but if main floods the channel faster than
// worker drains, the channel queue itself can grow. v0.1 trusts the
// worker drains fast enough; v0.2 should add an ack-based gating loop.

export interface PostMessageInputSetup {
  port:         MessagePort
  transferList: Transferable[]
  cleanup:      () => void
}

type RVFCMeta = { mediaTime: number }
type RVFCCallback = (now: number, meta: RVFCMeta) => void
type RVFCMethod = (cb: RVFCCallback) => number

export function setupPostMessageInput(inputStream: MediaStream): PostMessageInputSetup {
  // Hidden video element — never appended to DOM. Muted + playsInline so
  // autoplay isn't blocked by browser policies.
  const video = document.createElement('video')
  video.srcObject = inputStream
  video.muted     = true
  video.autoplay  = true
  video.playsInline = true

  // Don't await — play() can hang waiting for the consumer (us) to be
  // ready, which we already are. Failures are logged but non-fatal; rVFC
  // still fires once frames arrive.
  video.play().catch(err => {
    console.warn('[input-postmessage/main] video.play() rejected:', err)
  })

  const rvfc = (video as unknown as { requestVideoFrameCallback?: RVFCMethod })
    .requestVideoFrameCallback
    ?.bind(video)
  if (!rvfc) {
    throw new Error('setupPostMessageInput: requestVideoFrameCallback not supported on this browser')
  }

  const channel    = new MessageChannel()
  const mainPort   = channel.port1
  const workerPort = channel.port2

  let stopped = false

  const tick: RVFCCallback = (_now, meta) => {
    if (stopped) return
    try {
      // VideoFrame.timestamp is microseconds; rVFC's mediaTime is seconds.
      const frame = new VideoFrame(video, { timestamp: meta.mediaTime * 1_000_000 })
      mainPort.postMessage({ frame }, [frame])
    } catch (err) {
      // VideoFrame() can throw if the video isn't ready yet for this tick;
      // safe to skip — next rVFC will retry.
      console.warn('[input-postmessage/main] VideoFrame() failed; skipping frame:', err)
    }
    rvfc(tick)
  }
  rvfc(tick)

  const cleanup = () => {
    stopped = true
    mainPort.close()
    video.srcObject = null
    video.pause()
  }

  return {
    port:         workerPort,
    transferList: [workerPort],
    cleanup,
  }
}
