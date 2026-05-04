// Dispatches `topology.output` to the matching adapter and returns a
// uniform WritableStream<VideoFrame> that the input pipe can pipeTo.
//
//   'mstg'             → wires (transform.readable → InitData.outputWritable)
//                         internally; returns transform.writable as the sink
//                         so the caller's pipe is symmetric with the other
//                         two paths
//   'transfer-capture' → terminal sink that draws to renderer.canvas (which
//                         IS InitData.outputCanvas, transferred from main)
//   'bitmap-shuttle'   → terminal sink that ships ImageBitmaps via
//                         InitData.outputPort
//
// Hiding the MSTG two-stage pipe in here keeps handleInit's pipe wiring
// uniform: `inputReadable.pipeTo(outputSink, { signal })`.
//
// `signal` is honored by all internal pipes — when the worker shuts down
// (destroy command), aborting the signal cleanly tears down every pipe.

import type { InitData } from '../messages'
import type { Renderer } from './renderer'
import { createMstgOutput }            from './adapters/output_mstg'
import { createTransferCaptureOutput } from './adapters/output_transfer_capture'
import { createBitmapShuttleOutput }   from './adapters/output_bitmap_shuttle'

export function createOutputSink(
  data:     InitData,
  renderer: Renderer,
  signal:   AbortSignal,
): WritableStream<VideoFrame> {
  switch (data.topology.output) {
    case 'mstg': {
      if (!data.outputWritable) {
        throw new Error("create_output: topology.output='mstg' but InitData.outputWritable missing")
      }
      const transform = createMstgOutput(renderer)
      // Background pipe: transform → MSTG writable. handleInit will pipe its
      // input into transform.writable; backpressure flows through both.
      transform.readable.pipeTo(data.outputWritable, { signal }).catch(err => {
        if (!signal.aborted) console.warn('[create_output] mstg downstream pipe failed:', err)
      })
      return transform.writable
    }

    case 'transfer-capture':
      return createTransferCaptureOutput(renderer)

    case 'bitmap-shuttle': {
      if (!data.outputPort) {
        throw new Error("create_output: topology.output='bitmap-shuttle' but InitData.outputPort missing")
      }
      return createBitmapShuttleOutput(renderer, data.outputPort)
    }
  }
}
