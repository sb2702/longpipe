// Bitmap shuttle output adapter.
//
// Renderer draws to its OffscreenCanvas; we transfer-detach the bitmap
// and post it to main via a dedicated MessagePort. Main has a
// bitmaprenderer canvas + captureStream. One postMessage per frame; the
// bitmap is transferable so it's zero-copy on the wire.
//
// Universal fallback — works on every browser we tested.

import type { Renderer } from '../renderer'

export function createBitmapShuttleOutput(
  renderer: Renderer,
  port:     MessagePort,
  onFrame:  () => void = () => {},
): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    write(frame) {
      renderer.process(frame)
      frame.close()
      const bmp = renderer.canvas.transferToImageBitmap()
      port.postMessage({ bmp }, [bmp])
      onFrame()
    },
    close() {
      port.close()
    },
  })
}
