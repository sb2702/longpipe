// Main-side setup for the bitmap-shuttle output transport.
//
// Pairs with worker/adapters/output_bitmap_shuttle.ts on the worker side.
// Universal fallback path: works on every browser we tested (Chromium,
// Firefox, Safari). Also the path a transfer-capture (Safari) pipeline moves
// to after its worker dies, because a transferred canvas can't be taken back.
//
// What main thread owns:
//   - An off-DOM HTMLCanvasElement with bitmaprenderer context (the visible
//     surface that captureStream observes — must be on main since
//     OffscreenCanvas has no captureStream)
//   - A MessagePort the worker sends ImageBitmaps to per frame (a fresh one
//     per worker: attachWorker())
//   - The output MediaStreamTrack from canvas.captureStream()
//
// Per frame: worker → port.postMessage({bmp}) → ctx.transferFromImageBitmap.
// captureStream picks up the new bitmap automatically and emits it as a
// VideoFrame on the track.
//
// Raw passthrough (startRaw): main copies input frames to the canvas itself —
// during worker boot, with no GPU, and while a crashed worker restarts. It
// stops once the attached worker has delivered HANDOFF_AT_FRAME bitmaps.

import { createLogger } from '../debug'

const log = createLogger('bitmap-shuttle/main')

export interface BitmapShuttleOutputSetup {
  videoTrack:   MediaStreamTrack
  attachWorker: () => { port: MessagePort; transferList: Transferable[] }
  startRaw:     (inputStream: MediaStream) => boolean
  end:          () => void
  cleanup:      () => void
}

const DEFAULT_FPS = 30
// Worker's first few effect frames are slow (GPU shader compilation +
// pipeline warmup). If we hand off on the very first bitmap, the
// captureStream re-emits that single bitmap for ~200-300ms while the
// worker grinds out frame 2, looking like a freeze. Keep passthrough
// alive until the worker has produced HANDOFF_AT_FRAME bitmaps so the
// handoff happens at steady-state cadence.
const HANDOFF_AT_FRAME = 3

export function setupBitmapShuttleOutput(
  size: { w: number; h: number },
  fps:  number = DEFAULT_FPS,
): BitmapShuttleOutputSetup {
  // The canvas is created off-DOM (never appended). captureStream still
  // emits and the canvas is kept alive by references from the listener
  // closure (via ctx) and by the captureStream track's internal source
  // ref. Cleanup() drops both.
  const canvas = document.createElement('canvas')
  canvas.width  = size.w
  canvas.height = size.h

  const ctx = canvas.getContext('bitmaprenderer')
  if (!ctx) throw new Error('setupBitmapShuttleOutput: bitmaprenderer context not available')

  const stream = canvas.captureStream(fps)
  const tracks = stream.getVideoTracks()
  if (tracks.length === 0) throw new Error('setupBitmapShuttleOutput: captureStream produced no video tracks')
  const videoTrack = tracks[0]

  let mainPort: MessagePort | null = null
  let rawActive = false
  let rawVideo: HTMLVideoElement | null = null
  let workerFramesReceived = 0

  const stopRaw = () => {
    rawActive = false
    if (rawVideo) {
      rawVideo.srcObject = null
      rawVideo.pause()
      rawVideo = null
    }
  }

  const detach = () => {
    if (!mainPort) return
    mainPort.onmessage = null
    mainPort.close()
    mainPort = null
  }

  const attachWorker = () => {
    detach()
    const channel = new MessageChannel()
    const port = channel.port1
    mainPort = port
    workerFramesReceived = 0
    port.onmessage = (e: MessageEvent<{ bmp: ImageBitmap }>) => {
      if (mainPort !== port) { e.data.bmp.close(); return }
      if (rawActive) {
        workerFramesReceived++
        if (workerFramesReceived < HANDOFF_AT_FRAME) {
          e.data.bmp.close()
          return
        }
        stopRaw()
      }
      try {
        // The worker's output follows the input's orientation (a phone turned
        // mid-call): match the bitmap so captureStream emits its real size.
        const bmp = e.data.bmp
        if (bmp.width !== canvas.width || bmp.height !== canvas.height) {
          canvas.width  = bmp.width
          canvas.height = bmp.height
        }
        ctx.transferFromImageBitmap(bmp)
      } catch (err) {
        // transferFromImageBitmap can throw if the bitmap was somehow already
        // consumed; log + drop so the pipe doesn't stall.
        log.warn('transferFromImageBitmap failed:', err)
        e.data.bmp.close()
      }
    }
    port.start()
    return { port: channel.port2, transferList: [channel.port2] as Transferable[] }
  }

  const startRaw = (inputStream: MediaStream): boolean => {
    // A worker that is being replaced must not hand off (or draw) anymore.
    detach()
    stopRaw()
    if (!inputStream.getVideoTracks().some(t => t.readyState !== 'ended')) return false
    rawActive = true
    workerFramesReceived = 0

    const video = document.createElement('video')
    video.srcObject  = inputStream
    video.muted      = true
    video.playsInline = true
    void video.play().catch(() => {})
    rawVideo = video

    // Per-frame: createImageBitmap (with resize to canvas size for stable
    // captureStream output dims) → transferFromImageBitmap. Each iteration
    // schedules the next via rVFC (or rAF fallback for browsers without
    // rVFC on <video>; rare in our targets).
    const schedule: (cb: () => void) => void =
      'requestVideoFrameCallback' in video
        ? (cb) => { video.requestVideoFrameCallback(cb) }
        : (cb) => { requestAnimationFrame(cb) }

    const tick = () => {
      if (!rawActive || rawVideo !== video) return
      if (video.videoWidth === 0) {
        // input not ready yet — try again next frame
        schedule(tick)
        return
      }
      // Follow the input's (upright) orientation, as the worker does.
      const vw = video.videoWidth, vh = video.videoHeight
      if (vw !== vh && canvas.width !== canvas.height && (vw > vh) !== (canvas.width > canvas.height)) {
        const w = canvas.width
        canvas.width  = canvas.height
        canvas.height = w
      }
      createImageBitmap(video, {
        resizeWidth:   canvas.width,
        resizeHeight:  canvas.height,
        resizeQuality: 'medium',
      }).then(bmp => {
        if (!rawActive || rawVideo !== video) { bmp.close(); return }
        try { ctx.transferFromImageBitmap(bmp) } catch { bmp.close() }
        schedule(tick)
      }).catch(() => {
        // createImageBitmap can fail mid-stream (e.g., paused source).
        // Don't propagate — just retry.
        schedule(tick)
      })
    }
    schedule(tick)
    return true
  }

  let ended = false
  const end = () => {
    if (ended) return
    ended = true
    stopRaw()
    detach()
    videoTrack.stop()
  }

  return {
    videoTrack,
    attachWorker,
    startRaw,
    end,
    cleanup: end,
  }
}
