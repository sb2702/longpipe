// Main-side setup for the MediaStreamTrackGenerator output transport.
//
// Pairs with worker/adapters/output_mstg.ts. Chrome-only path.
//
// What main thread owns:
//   - A MediaStreamTrackGenerator (kind: 'video'). Its `writable` is a
//     WritableStream<VideoFrame> that's transferable. The generator IS
//     a MediaStreamTrack — captureStream isn't involved.
//
// We transfer the writable to the worker; the track stays on main and is
// returned for inclusion in the output MediaStream. No canvas, no
// per-frame messaging on main.

export interface MstgOutputSetup {
  videoTrack:   MediaStreamTrack
  writable:     WritableStream<VideoFrame>
  transferList: Transferable[]
  cleanup:      () => void
}

// MediaStreamTrackGenerator isn't in default lib.dom yet; declare the
// shape we use to avoid a global type augmentation.
type MstgInstance = MediaStreamTrack & { writable: WritableStream<VideoFrame> }
type MstgCtor = new (init: { kind: 'video' }) => MstgInstance

export function setupMstgOutput(): MstgOutputSetup {
  const Ctor = (self as unknown as { MediaStreamTrackGenerator?: MstgCtor }).MediaStreamTrackGenerator
  if (!Ctor) throw new Error('setupMstgOutput: MediaStreamTrackGenerator not supported on this browser')

  const generator = new Ctor({ kind: 'video' })
  const writable  = generator.writable

  const cleanup = () => {
    // Aborting the writable signals upstream that we're done. The track
    // itself is also stopped so the output MediaStream cleanly ends.
    void writable.abort().catch(() => {})
    generator.stop()
  }

  return {
    videoTrack:   generator,
    writable,
    transferList: [writable],
    cleanup,
  }
}
