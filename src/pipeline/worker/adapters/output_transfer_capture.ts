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

export function createTransferCaptureOutput(renderer: Renderer): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    write(frame) {
      renderer.process(frame)
      frame.close()
    },
  })
}
