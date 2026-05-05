// Main-side output dispatcher. Mirrors worker/create_output.ts: switches
// on `topology.output` and returns a unified shape Pipeline can plumb
// into InitData fields + transferList. Also exposes the output videoTrack
// (for inclusion in the public Pipeline.stream) and an optional
// startPassthrough hook (only bitmap-shuttle supports it for now).

import type { OutputPath } from './topology'
import { setupBitmapShuttleOutput }   from './transports/output_bitmap_shuttle'
import { setupMstgOutput }            from './transports/output_mstg'
import { setupTransferCaptureOutput } from './transports/output_transfer_capture'

export interface OutputSetup {
  videoTrack:       MediaStreamTrack
  initFields: {
    outputWritable?: WritableStream<VideoFrame>
    outputCanvas?:   OffscreenCanvas
    outputPort?:     MessagePort
  }
  transferList:     Transferable[]
  // Live passthrough during worker boot. Only bitmap-shuttle implements
  // it today (the worker-owned topologies write directly to the output
  // surface so there's nowhere for main to inject passthrough frames).
  startPassthrough?: (inputStream: MediaStream) => void
  cleanup:          () => void
}

export function setupOutput(path: OutputPath, size: { w: number; h: number }): OutputSetup {
  switch (path) {
    case 'mstg': {
      const s = setupMstgOutput()
      return {
        videoTrack:   s.videoTrack,
        initFields:   { outputWritable: s.writable },
        transferList: s.transferList,
        cleanup:      s.cleanup,
      }
    }
    case 'transfer-capture': {
      const s = setupTransferCaptureOutput(size)
      return {
        videoTrack:   s.videoTrack,
        initFields:   { outputCanvas: s.canvas },
        transferList: s.transferList,
        cleanup:      s.cleanup,
      }
    }
    case 'bitmap-shuttle': {
      const s = setupBitmapShuttleOutput(size)
      return {
        videoTrack:       s.videoTrack,
        initFields:       { outputPort: s.port },
        transferList:     s.transferList,
        startPassthrough: s.startPassthrough,
        cleanup:          s.cleanup,
      }
    }
  }
}
