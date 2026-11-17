// Main-side setup for the MediaStreamTrackGenerator output transport.
//
// Pairs with worker/adapters/output_mstg.ts. Chromium path.
//
// What main thread owns:
//   - A MediaStreamTrackGenerator (kind: 'video'). The generator IS the
//     output MediaStreamTrack; it never changes for the pipeline's lifetime.
//   - A tiny relay worker (relay_worker.ts) that holds the generator's
//     transferred `writable` and writes either raw camera frames or the GPU
//     worker's frames into it. That is what lets the SAME track carry raw
//     video when the GPU worker is booting, has no GPU, lost its device, or
//     crashed and is being restarted.
//
// No canvas and no per-frame work on the main thread.

import { createRelayWorker } from './relay_worker'
import { createLogger } from '../debug'

const log = createLogger('mstg/main')

export interface MstgOutputSetup {
  videoTrack:   MediaStreamTrack
  // New frame port for a (re)started GPU worker. The relay keeps writing raw
  // frames (if raw is active) until this port delivers its 3rd frame.
  attachWorker: () => { port: MessagePort; transferList: Transferable[] }
  // Raw passthrough from the input track, starting now. Detaches the current
  // GPU-worker port (it is dead or about to be).
  startRaw:     (inputStream: MediaStream) => boolean
  end:          () => void
  cleanup:      () => void
}

type MstgInstance = MediaStreamTrack & { writable: WritableStream<VideoFrame> }
type MstgCtor = new (init: { kind: 'video' }) => MstgInstance
type MstpCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

export function setupMstgOutput(onRelayMode?: (mode: 'raw' | 'worker') => void): MstgOutputSetup {
  const Ctor = (self as unknown as { MediaStreamTrackGenerator?: MstgCtor }).MediaStreamTrackGenerator
  if (!Ctor) throw new Error('setupMstgOutput: MediaStreamTrackGenerator not supported on this browser')

  const generator = new Ctor({ kind: 'video' })
  const relay = createRelayWorker()
  relay.postMessage({ type: 'init', writable: generator.writable }, [generator.writable])
  relay.onmessage = (e: MessageEvent<{ type: string; mode?: 'raw' | 'worker' }>) => {
    if (e.data.type === 'mode' && e.data.mode) {
      log('relay mode:', e.data.mode)
      onRelayMode?.(e.data.mode)
    }
  }
  relay.onerror = (e) => console.error('[longpipe/pipeline] output relay worker error:', e.message)

  const attachWorker = () => {
    const ch = new MessageChannel()
    relay.postMessage({ type: 'worker', port: ch.port1 }, [ch.port1])
    return { port: ch.port2, transferList: [ch.port2] as Transferable[] }
  }

  // The relay keeps ONE camera reader per input track (see relay_worker.ts):
  // a restart re-enables it instead of transferring a fresh readable.
  let rawTrack: MediaStreamTrack | null = null
  const startRaw = (inputStream: MediaStream): boolean => {
    const track = inputStream.getVideoTracks()[0]
    const Proc = (self as unknown as { MediaStreamTrackProcessor?: MstpCtor }).MediaStreamTrackProcessor
    if (!track || track.readyState === 'ended' || !Proc) return false
    if (track === rawTrack) {
      relay.postMessage({ type: 'raw', detach: true })
      return true
    }
    // A second processor on the camera track (the GPU worker holds the
    // first). Cancelling it detaches only this processor — the camera track
    // itself is untouched.
    const { readable } = new Proc({ track })
    rawTrack = track
    relay.postMessage({ type: 'raw', readable, detach: true }, [readable])
    return true
  }

  let ended = false
  let closed: Promise<void> | null = null
  const end = () => {
    if (ended) return
    ended = true
    // The relay releases its camera reader and the generator's writer, then
    // acknowledges; cleanup() waits for that before terminating it.
    closed = new Promise<void>((resolve) => {
      const t = setTimeout(resolve, 1000)
      relay.addEventListener('message', (e: MessageEvent<{ type: string }>) => {
        if (e.data.type === 'closed') { clearTimeout(t); resolve() }
      })
    })
    relay.postMessage({ type: 'close' })
    generator.stop()
  }

  let terminated = false
  const cleanup = () => {
    end()
    if (terminated) return
    terminated = true
    void closed!.then(() => relay.terminate())
  }

  return { videoTrack: generator, attachWorker, startRaw, end, cleanup }
}
