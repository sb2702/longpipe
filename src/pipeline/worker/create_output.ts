// Dispatches `topology.output` to the matching adapter and returns a
// uniform WritableStream<VideoFrame> that the input pipe can pipeTo.
//
//   'mstg'             → sink that posts each rendered VideoFrame to the
//                         output relay worker via InitData.outputFramePort
//   'transfer-capture' → terminal sink that draws to renderer.canvas (which
//                         IS InitData.outputCanvas, transferred from main)
//   'bitmap-shuttle'   → terminal sink that ships ImageBitmaps via
//                         InitData.outputPort
//
// `onFrame` counts frames written to the output (heartbeat → main watchdog).
//
// createRawOutputSink() is the no-GPU variant: on transfer-capture the worker
// owns the only drawable surface, so it draws raw frames with a 2D context.
// The other paths do raw passthrough on the main side (relay / main canvas).

import type { InitData } from '../messages'
import type { Renderer } from './renderer'
import { createMstgOutput }            from './adapters/output_mstg'
import { createTransferCaptureOutput } from './adapters/output_transfer_capture'
import { createBitmapShuttleOutput }   from './adapters/output_bitmap_shuttle'

export function createOutputSink(
  data:     InitData,
  renderer: Renderer,
  onFrame:  () => void,
): WritableStream<VideoFrame> {
  switch (data.topology.output) {
    case 'mstg': {
      if (!data.outputFramePort) {
        throw new Error("create_output: topology.output='mstg' but InitData.outputFramePort missing")
      }
      return createMstgOutput(renderer, data.outputFramePort, onFrame)
    }

    case 'transfer-capture':
      return createTransferCaptureOutput(renderer, onFrame)

    case 'bitmap-shuttle': {
      if (!data.outputPort) {
        throw new Error("create_output: topology.output='bitmap-shuttle' but InitData.outputPort missing")
      }
      return createBitmapShuttleOutput(renderer, data.outputPort, onFrame)
    }
  }
}

// Raw (no GPU) sink for transfer-capture. Returns null when the canvas can't
// take a 2D context (a failed WebGPU/WebGL attempt may already own it) — main
// then moves the output to a canvas it controls.
export function createRawOutputSink(data: InitData, onFrame: () => void): WritableStream<VideoFrame> | null {
  if (data.topology.output !== 'transfer-capture' || !data.outputCanvas) return null
  const canvas = data.outputCanvas
  let ctx: OffscreenCanvasRenderingContext2D | null = null
  try { ctx = canvas.getContext('2d') } catch { ctx = null }
  if (!ctx) return null
  const c = ctx
  return new WritableStream<VideoFrame>({
    write(frame) {
      try { c.drawImage(frame, 0, 0, canvas.width, canvas.height) } finally { frame.close() }
      onFrame()
    },
  })
}
