// transfer-capture output adapter.
//
// Renderer's canvas IS the captured canvas (transferControlToOffscreen on
// main, transferred to worker). The renderer writes directly to it; main-
// thread captureStream observes. No per-frame messaging.
//
// Pipe: inputReadable → captureSink (terminal). The captureSink consumes
// each frame, runs the renderer, closes the source frame, and resolves —
// no downstream queue.

import type { Renderer } from '../renderer'
import { frameEntryTime } from '../latency'

export function createTransferCaptureOutput(renderer: Renderer, onFrame: (entry?: number) => void = () => {}): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    async write(frame) {
      const entry = frameEntryTime(frame, performance.now())
      try {
        const src = await renderer.prepareSource(frame)
        try { renderer.process(frame, src) } finally { if (src !== frame) src.close() }
        renderer.markSubmitted()
      } finally {
        frame.close()
      }
      onFrame(entry)
      // captureStream samples the canvas and never pushes back, so, as on
      // bitmap-shuttle, pace on the GPU: at most one frame in flight.
      await renderer.waitForGpu(1)
    },
  })
}
