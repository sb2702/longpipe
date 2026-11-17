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

export function createMstgOutput(renderer: Renderer, port: MessagePort, onFrame: () => void): WritableStream<VideoFrame> {
  return new WritableStream<VideoFrame>({
    write(frame) {
      renderer.process(frame)
      const out = new VideoFrame(renderer.canvas, { timestamp: frame.timestamp })
      frame.close()
      port.postMessage({ frame: out }, [out])
      onFrame()
    },
    close() {
      port.close()
    },
  })
}
