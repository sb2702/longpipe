// MSTG output adapter.
//
// Worker renders to its own OffscreenCanvas, wraps each rendered frame in a
// new VideoFrame, and posts it (transferred, zero-copy) to the output relay
// worker, which owns the MediaStreamTrackGenerator's writable (see
// transports/relay_worker.ts for why the writable no longer lives here).
//
// Pipe: inputReadable → this sink. The relay drops a frame if the generator
// hasn't accepted the previous one yet, so there's no queue to grow.

import type { Renderer } from '../renderer'
import { frameEntryTime } from '../latency'

export function createMstgOutput(renderer: Renderer, port: MessagePort, onFrame: (entry?: number) => void): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    async write(frame) {
      const entry = frameEntryTime(frame, performance.now())
      let out: VideoFrame
      try {
        const src = await renderer.prepareSource(frame)
        try { renderer.process(frame, src) } finally { if (src !== frame) src.close() }
        out = new VideoFrame(renderer.canvas, { timestamp: frame.timestamp })
      } finally {
        // Closed even when processing throws (GPU lost mid-frame): an unclosed
        // camera frame pins a capture buffer until GC.
        frame.close()
      }
      port.postMessage({ frame: out }, [out])
      onFrame(entry)
    },
    close() {
      port.close()
    },
  })
}
