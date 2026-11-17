// Main-side input dispatcher. Mirrors worker/create_input.ts: switches on
// `topology.input` and returns a unified shape Pipeline can plumb into
// InitData fields + transferList.

import type { InputPath } from './topology'
import { setupPostMessageInput } from './transports/input_postmessage'
import { setupMstpInput }        from './transports/input_mstp'

export interface InputSetup {
  // Fields to merge into InitData (presence depends on path).
  initFields: {
    inputReadable?: ReadableStream<VideoFrame>
    inputPort?:     MessagePort
  }
  transferList: Transferable[]
  // Input frames delivered so far, when this path can count them (rVFC).
  framesIn?:    () => number
  cleanup:      () => void
}

export function setupInput(path: InputPath, inputStream: MediaStream): InputSetup {
  switch (path) {
    case 'mstp': {
      const s = setupMstpInput(inputStream)
      return {
        initFields:   { inputReadable: s.readable },
        transferList: s.transferList,
        cleanup:      s.cleanup,
      }
    }
    case 'rvfc-postmessage': {
      const s = setupPostMessageInput(inputStream)
      return {
        initFields:   { inputPort: s.port },
        transferList: s.transferList,
        framesIn:     s.framesIn,
        cleanup:      s.cleanup,
      }
    }
  }
}
