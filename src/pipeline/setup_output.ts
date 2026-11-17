// Main-side output dispatcher. Mirrors worker/create_output.ts: switches
// on `topology.output` and returns a unified OutputHandle the Pipeline uses
// for the whole life of the output track:
//
//   attachWorker()  wire a (new) GPU worker → InitData fields + transferList
//   startRaw()      main-side raw passthrough on the SAME track, until the
//                   next attached worker takes over (boot, no GPU, GPU loss,
//                   worker restart)
//   end()           the input ended → stop the output track
//
// transfer-capture is the exception: its canvas is transferred to ONE worker
// and can't come back, so it has no main-side raw path and can't take a
// second worker (`reusable: false`). When that worker dies, the Pipeline
// replaces it with a bitmap-shuttle output (a new track) — see
// Pipeline.takeoverOutput().

import type { OutputPath } from './topology'
import type { InitData } from './messages'
import { setupBitmapShuttleOutput }   from './transports/output_bitmap_shuttle'
import { setupMstgOutput }            from './transports/output_mstg'
import { setupTransferCaptureOutput } from './transports/output_transfer_capture'

export interface OutputHandle {
  path:         OutputPath
  videoTrack:   MediaStreamTrack
  // false → attachWorker() works once only (transfer-capture).
  reusable:     boolean
  attachWorker: () => { initFields: Partial<InitData>; transferList: Transferable[] }
  // false when this path can't do main-side raw (transfer-capture).
  startRaw:     (inputStream: MediaStream) => boolean
  end:          () => void
  cleanup:      () => void
}

export function setupOutput(path: OutputPath, size: { w: number; h: number }): OutputHandle {
  switch (path) {
    case 'mstg': {
      const s = setupMstgOutput()
      return {
        path,
        videoTrack:   s.videoTrack,
        reusable:     true,
        attachWorker: () => {
          const a = s.attachWorker()
          return { initFields: { outputFramePort: a.port }, transferList: a.transferList }
        },
        startRaw:     s.startRaw,
        end:          s.end,
        cleanup:      s.cleanup,
      }
    }
    case 'transfer-capture': {
      const s = setupTransferCaptureOutput(size)
      let attached = false
      return {
        path,
        videoTrack:   s.videoTrack,
        reusable:     false,
        attachWorker: () => {
          if (attached) throw new Error('transfer-capture: output canvas already belongs to a worker')
          attached = true
          return { initFields: { outputCanvas: s.canvas }, transferList: s.transferList }
        },
        startRaw:     () => false,
        end:          s.cleanup,
        cleanup:      s.cleanup,
      }
    }
    case 'bitmap-shuttle': {
      const s = setupBitmapShuttleOutput(size)
      return {
        path,
        videoTrack:   s.videoTrack,
        reusable:     true,
        attachWorker: () => {
          const a = s.attachWorker()
          return { initFields: { outputPort: a.port }, transferList: a.transferList }
        },
        startRaw:     s.startRaw,
        end:          s.end,
        cleanup:      s.cleanup,
      }
    }
  }
}
