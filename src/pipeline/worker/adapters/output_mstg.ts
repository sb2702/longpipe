// MSTG output adapter.
//
// Worker renders to its own OffscreenCanvas, wraps each rendered frame in
// a new VideoFrame, and writes it into a transferred WritableStream
// constructed on main thread from a MediaStreamTrackGenerator.
//
// Pipe: inputReadable → rendererTransform → mstgWritable.
// Backpressure flows up the chain automatically.

import type { Renderer } from '../renderer'

export function createMstgOutput(renderer: Renderer): TransformStream<VideoFrame, VideoFrame> {
  return new TransformStream<VideoFrame, VideoFrame>({
    transform(frame, controller) {
      renderer.process(frame)
      const out = new VideoFrame(renderer.canvas, { timestamp: frame.timestamp })
      frame.close()
      controller.enqueue(out)
    },
  })
}
