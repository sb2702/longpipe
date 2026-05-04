// Wraps the input → output Streams pipe with first-frame detection and
// abort handling. Returns a Promise that resolves when the pipe completes
// cleanly (input ended), rejects on error, and short-circuits on abort.
//
// `onFirstFrame` fires the moment the first VideoFrame successfully
// traverses the pipe — i.e., the renderer's process() returned without
// throwing and the frame was enqueued toward the sink. That's the signal
// for handleInit to emit the 'ready' event back to main.
//
// Implementation: insert a tiny TransformStream tap between input and
// sink. Tap's transform() fires `onFirstFrame` once. Backpressure flows
// through the tap unchanged.

export interface PipeOptions {
  input:        ReadableStream<VideoFrame>
  output:       WritableStream<VideoFrame>
  signal:       AbortSignal
  onFirstFrame: () => void
}

export function startPipe(opts: PipeOptions): Promise<void> {
  let firstFrameSeen = false

  const tap = new TransformStream<VideoFrame, VideoFrame>({
    transform(frame, controller) {
      controller.enqueue(frame)
      if (!firstFrameSeen) {
        firstFrameSeen = true
        // Defer the callback so onFirstFrame's side effects (postMessage,
        // etc.) don't run inside the transform's microtask and risk
        // re-entrancy with the next frame's enqueue.
        queueMicrotask(opts.onFirstFrame)
      }
    },
  })

  return opts.input
    .pipeThrough(tap, { signal: opts.signal })
    .pipeTo(opts.output, { signal: opts.signal })
}
