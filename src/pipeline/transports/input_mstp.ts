// Main-side setup for the MediaStreamTrackProcessor input transport.
//
// Pairs with worker/adapters/input_mstp.ts. Chrome-only path (Firefox /
// Safari fall back to rvfc-postmessage).
//
// What main thread owns:
//   - A MediaStreamTrackProcessor wrapping the input video track. Its
//     `readable` is a ReadableStream<VideoFrame> that's transferable.
//
// We transfer the readable to the worker — main does no per-frame work
// for input. MSTP itself drops late frames when the reader can't keep up,
// matching videoconf-correct backpressure.

export interface MstpInputSetup {
  readable:     ReadableStream<VideoFrame>
  transferList: Transferable[]
  cleanup:      () => void
}

// MediaStreamTrackProcessor isn't in default lib.dom yet; declare the
// shape we use to avoid a global type augmentation.
type MstpCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

export function setupMstpInput(inputStream: MediaStream): MstpInputSetup {
  const tracks = inputStream.getVideoTracks()
  if (tracks.length === 0) throw new Error('setupMstpInput: input MediaStream has no video tracks')
  const track = tracks[0]

  const Ctor = (self as unknown as { MediaStreamTrackProcessor?: MstpCtor }).MediaStreamTrackProcessor
  if (!Ctor) throw new Error('setupMstpInput: MediaStreamTrackProcessor not supported on this browser')

  const processor = new Ctor({ track })
  const readable  = processor.readable

  const cleanup = () => {
    // Cancel the readable to signal upstream that we're done. Track itself
    // belongs to the caller's MediaStream — we don't stop() it.
    void readable.cancel().catch(() => {})
  }

  return {
    readable,
    transferList: [readable],
    cleanup,
  }
}
