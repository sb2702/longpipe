// Main-side setup for the bitmap-shuttle output transport.
//
// Pairs with worker/adapters/output_bitmap_shuttle.ts on the worker side.
// Universal fallback path: works on every browser we tested (Chromium,
// Firefox, Safari).
//
// What main thread owns:
//   - An off-DOM HTMLCanvasElement with bitmaprenderer context (the visible
//     surface that captureStream observes — must be on main since
//     OffscreenCanvas has no captureStream)
//   - A MessagePort the worker sends ImageBitmaps to per frame
//   - The output MediaStreamTrack from canvas.captureStream()
//
// Per frame: worker → port.postMessage({bmp}) → ctx.transferFromImageBitmap.
// captureStream picks up the new bitmap automatically and emits it as a
// VideoFrame on the track.

import { createLogger } from '../debug'

const log = createLogger('bitmap-shuttle/main')

export interface BitmapShuttleOutputSetup {
  videoTrack:   MediaStreamTrack
  port:         MessagePort           // hand this to worker via transfer list
  transferList: Transferable[]
  // Stream input frames straight to the output canvas while the worker is
  // initializing (autotune + weight fetch + first frame can take 1-3s).
  // Auto-stops the moment the worker posts its first bitmap so the handoff
  // is seamless. Safe to call once; no-op on subsequent calls.
  startPassthrough: (inputStream: MediaStream) => void
  cleanup:      () => void
}

const DEFAULT_FPS = 30

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

  const channel    = new MessageChannel()
  const mainPort   = channel.port1
  const workerPort = channel.port2

  // Passthrough state — only relevant between startPassthrough() and the
  // worker reaching steady state. Closure-captured so it can be torn down
  // from either trigger point.
  let passthroughActive = false
  let passthroughVideo: HTMLVideoElement | null = null
  let stopPassthrough: () => void = () => {}

  // Worker's first few effect frames are slow (GPU shader compilation +
  // pipeline warmup). If we hand off on the very first bitmap, the
  // captureStream re-emits that single bitmap for ~200-300ms while the
  // worker grinds out frame 2, looking like a freeze. Keep passthrough
  // alive until the worker has produced HANDOFF_AT_FRAME bitmaps so the
  // handoff happens at steady-state cadence.
  const HANDOFF_AT_FRAME = 3
  let workerFramesReceived = 0

  mainPort.onmessage = (e: MessageEvent<{ bmp: ImageBitmap }>) => {
    if (passthroughActive) {
      workerFramesReceived++
      if (workerFramesReceived < HANDOFF_AT_FRAME) {
        e.data.bmp.close()
        return
      }
      stopPassthrough()
    }
    try {
      ctx.transferFromImageBitmap(e.data.bmp)
    } catch (err) {
      // transferFromImageBitmap can throw if the bitmap was somehow already
      // consumed; log + drop so the pipe doesn't stall.
      log.warn('transferFromImageBitmap failed:', err)
    }
  }
  mainPort.start()

  const startPassthrough = (inputStream: MediaStream) => {
    if (passthroughActive) return
    passthroughActive = true

    const video = document.createElement('video')
    video.srcObject  = inputStream
    video.muted      = true
    video.playsInline = true
    void video.play().catch(() => {})
    passthroughVideo = video

    // Per-frame: createImageBitmap (with resize to canvas size for stable
    // captureStream output dims) → transferFromImageBitmap. Each iteration
    // schedules the next via rVFC (or rAF fallback for browsers without
    // rVFC on <video>; rare in our targets).
    const schedule: (cb: () => void) => void =
      'requestVideoFrameCallback' in video
        ? (cb) => { video.requestVideoFrameCallback(cb) }
        : (cb) => { requestAnimationFrame(cb) }

    const tick = () => {
      if (!passthroughActive) return
      if (video.videoWidth === 0) {
        // input not ready yet — try again next frame
        schedule(tick)
        return
      }
      createImageBitmap(video, {
        resizeWidth:   canvas.width,
        resizeHeight:  canvas.height,
        resizeQuality: 'medium',
      }).then(bmp => {
        if (!passthroughActive) { bmp.close(); return }
        try { ctx.transferFromImageBitmap(bmp) } catch { /* drop */ }
        schedule(tick)
      }).catch(() => {
        // createImageBitmap can fail mid-stream (e.g., paused source).
        // Don't propagate — just retry.
        schedule(tick)
      })
    }
    schedule(tick)

    stopPassthrough = () => {
      passthroughActive = false
      if (passthroughVideo) {
        passthroughVideo.srcObject = null
        passthroughVideo.pause()
        passthroughVideo = null
      }
    }
  }

  const cleanup = () => {
    stopPassthrough()
    mainPort.onmessage = null
    mainPort.close()
    videoTrack.stop()
  }

  return {
    videoTrack,
    port:         workerPort,
    transferList: [workerPort],
    startPassthrough,
    cleanup,
  }
}
