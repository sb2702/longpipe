// Main-side setup for the transfer-capture output transport.
//
// Pairs with worker/adapters/output_transfer_capture.ts. Chrome + Safari
// (Firefox throws on captureStream() after transferControlToOffscreen).
//
// What main thread owns:
//   - An off-DOM HTMLCanvasElement. We call captureStream() FIRST (while
//     main still has a 2D/null context, or none) to bind the track to the
//     canvas, THEN transferControlToOffscreen() to hand drawing to the
//     worker. Order matters: once transferred, captureStream() throws on
//     the original canvas.
//
// Per frame: worker draws directly to its OffscreenCanvas; the bound
// captureStream emits the new frame on the track automatically. No
// MessagePort, no postMessage — pure GPU → captureStream observation.

export interface TransferCaptureOutputSetup {
  videoTrack:   MediaStreamTrack
  canvas:       OffscreenCanvas      // hand to worker via transfer list
  transferList: Transferable[]
  cleanup:      () => void
}

const DEFAULT_FPS = 30

export function setupTransferCaptureOutput(
  size: { w: number; h: number },
  fps:  number = DEFAULT_FPS,
): TransferCaptureOutputSetup {
  const canvas = document.createElement('canvas')
  canvas.width  = size.w
  canvas.height = size.h

  // captureStream BEFORE transferControlToOffscreen — the track must be
  // bound to the canvas while main still controls it. Reversing the order
  // throws InvalidStateError on Chrome.
  const stream = canvas.captureStream(fps)
  const tracks = stream.getVideoTracks()
  if (tracks.length === 0) throw new Error('setupTransferCaptureOutput: captureStream produced no video tracks')
  const videoTrack = tracks[0]

  // Now hand drawing to the worker.
  const offscreen = canvas.transferControlToOffscreen()

  const cleanup = () => {
    videoTrack.stop()
  }

  return {
    videoTrack,
    canvas:       offscreen,
    transferList: [offscreen],
    cleanup,
  }
}
