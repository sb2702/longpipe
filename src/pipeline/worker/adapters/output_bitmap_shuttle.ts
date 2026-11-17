// Bitmap shuttle output adapter.
//
// Renderer draws to its OffscreenCanvas; we transfer-detach the bitmap
// and post it to main via a dedicated MessagePort. Main has a
// bitmaprenderer canvas + captureStream. One postMessage per frame; the
// bitmap is transferable so it's zero-copy on the wire.
//
// Universal fallback — works on every browser we tested.

import type { Renderer } from '../renderer'
import { frameEntryTime } from '../latency'

export function createBitmapShuttleOutput(
  renderer: Renderer,
  port:     MessagePort,
  onFrame:  (entry?: number) => void = () => {},
): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    async write(frame) {
      const entry = frameEntryTime(frame, performance.now())
      const src = await renderer.prepareSource(frame)
      try { renderer.process(frame, src) } finally { if (src !== frame) src.close() }
      renderer.markSubmitted()
      frame.close()
      const bmp = renderer.canvas.transferToImageBitmap()
      port.postMessage({ bmp }, [bmp])
      onFrame(entry)
      // Firefox (this path) + WebGPU: don't take the next frame until the GPU has
      // finished this one, so latency can't build up (stale input frames drop instead).
      await renderer.waitForGpu(1)
    },
    close() {
      port.close()
    },
  })
}
