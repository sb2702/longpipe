// MSTP input adapter — receives a transferred ReadableStream<VideoFrame>
// constructed on main thread from a MediaStreamTrackProcessor.
//
// MSTP itself drops late frames when the reader can't keep up (videoconf-
// correct backpressure). The worker just pumps from the readable.

export function createMstpInput(readable: ReadableStream<VideoFrame>): ReadableStream<VideoFrame> {
  return readable
}
