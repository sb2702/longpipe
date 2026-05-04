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

export interface BitmapShuttleOutputSetup {
  videoTrack:   MediaStreamTrack
  port:         MessagePort           // hand this to worker via transfer list
  transferList: Transferable[]
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

  mainPort.onmessage = (e: MessageEvent<{ bmp: ImageBitmap }>) => {
    try {
      ctx.transferFromImageBitmap(e.data.bmp)
    } catch (err) {
      // transferFromImageBitmap can throw if the bitmap was somehow already
      // consumed; log + drop so the pipe doesn't stall.
      console.warn('[bitmap-shuttle/main] transferFromImageBitmap failed:', err)
    }
  }
  mainPort.start()

  const cleanup = () => {
    mainPort.onmessage = null
    mainPort.close()
    videoTrack.stop()
  }

  return {
    videoTrack,
    port:         workerPort,
    transferList: [workerPort],
    cleanup,
  }
}
