// Wraps the input → output Streams pipe with abort handling. Returns a
// Promise that resolves when the pipe completes cleanly (input ended),
// rejects on error, and short-circuits on abort.
//
// `onFirstFrame` is optional — used when callers want a tap that fires
// once on the first VideoFrame to traverse the pipe. Currently unused
// (handleStartRender emits 'ready' directly after attaching the network
// rather than waiting for a frame), but kept for future hooks.

export interface PipeOptions {
  input:         ReadableStream<VideoFrame>
  output:        WritableStream<VideoFrame>
  signal:        AbortSignal
  onFirstFrame?: () => void
}

export function startPipe(opts: PipeOptions): Promise<void> {
  if (!opts.onFirstFrame) {
    return opts.input.pipeTo(opts.output, { signal: opts.signal })
  }

  let firstFrameSeen = false
  const onFirstFrame = opts.onFirstFrame
  const tap = new TransformStream<VideoFrame, VideoFrame>({
    transform(frame, controller) {
      controller.enqueue(frame)
      if (!firstFrameSeen) {
        firstFrameSeen = true
        // Defer the callback so onFirstFrame's side effects (postMessage,
        // etc.) don't run inside the transform's microtask and risk
        // re-entrancy with the next frame's enqueue.
        queueMicrotask(onFirstFrame)
      }
    },
  })

  return opts.input
    .pipeThrough(tap, { signal: opts.signal })
    .pipeTo(opts.output, { signal: opts.signal })
}
