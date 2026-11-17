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
// `onFrame(entry)` counts frames written to the output (heartbeat → main
// watchdog) and gets the frame's entry time for getStats().latencyMs (latency.ts).
//
// createRawOutputSink() is the no-GPU variant: on transfer-capture the worker
// owns the only drawable surface, so it draws raw frames with a 2D context.
// The other paths do raw passthrough on the main side (relay / main canvas).

import type { InitData } from '../messages'
import { frameResidualOrientation, uprightSize, isIdentity } from '~/model/orientation.ts'
import type { Renderer } from './renderer'
import { frameEntryTime } from './latency'
import { createMstgOutput }            from './adapters/output_mstg'
import { createTransferCaptureOutput } from './adapters/output_transfer_capture'
import { createBitmapShuttleOutput }   from './adapters/output_bitmap_shuttle'

export function createOutputSink(
  data:     InitData,
  renderer: Renderer,
  onFrame:  (entry?: number) => void,
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
export function createRawOutputSink(data: InitData, onFrame: (entry?: number) => void): WritableStream<VideoFrame> | null {
  if (data.topology.output !== 'transfer-capture' || !data.outputCanvas) return null
  const canvas = data.outputCanvas
  let ctx: OffscreenCanvasRenderingContext2D | null = null
  try { ctx = canvas.getContext('2d') } catch { ctx = null }
  if (!ctx) return null
  const c = ctx
  return new WritableStream<VideoFrame>({
    write(frame) {
      const entry = frameEntryTime(frame, performance.now())
      try { drawOriented(c, canvas, frame) } finally { frame.close() }
      onFrame(entry)
    },
  })
}

// Draw a frame upright: apply the hidden orientation main measured (Safari
// camera frames; see orientation_probe.ts) and follow its portrait/landscape
// shape, like the GPU renderer does.
function drawOriented(c: OffscreenCanvasRenderingContext2D, canvas: OffscreenCanvas, frame: VideoFrame): void {
  const o = frameResidualOrientation(frame)
  const up = uprightSize(frame.displayWidth, frame.displayHeight, o)
  if (up.w !== up.h && canvas.width !== canvas.height && (up.w > up.h) !== (canvas.width > canvas.height)) {
    const w = canvas.width
    canvas.width  = canvas.height
    canvas.height = w
  }
  const W = canvas.width, H = canvas.height
  if (isIdentity(o)) { c.drawImage(frame, 0, 0, W, H); return }
  c.save()
  // Last transform applies first to the content: rotate clockwise, then flip.
  if (o.flip) { c.translate(W, 0); c.scale(-1, 1) }
  c.translate(W / 2, H / 2)
  c.rotate(o.rotation * Math.PI / 180)
  const sideways = o.rotation === 90 || o.rotation === 270
  const sw = sideways ? H : W, sh = sideways ? W : H
  c.drawImage(frame, -sw / 2, -sh / 2, sw, sh)
  c.restore()
}
