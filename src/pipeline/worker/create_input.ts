// Dispatches `topology.input` to the matching adapter and returns a
// ReadableStream<VideoFrame> the renderer's pipe can consume from.
//
//   'mstp'             → uses InitData.inputReadable (transferred from a
//                         MediaStreamTrackProcessor on main)
//   'rvfc-postmessage' → uses InitData.inputPort (a MessagePort that main
//                         posts each new VideoFrame to via rVFC)
//
// Both paths produce the same ReadableStream<VideoFrame> shape, so
// downstream (Renderer + output adapters) is transport-agnostic.

import type { InitData } from '../messages'
import { createMstpInput }        from './adapters/input_mstp'
import { createPostMessageInput } from './adapters/input_postmessage'

export function createInputStream(data: InitData): ReadableStream<VideoFrame> {
  switch (data.topology.input) {
    case 'mstp':
      if (!data.inputReadable) {
        throw new Error("create_input: topology.input='mstp' but InitData.inputReadable missing")
      }
      return createMstpInput(data.inputReadable)
    case 'rvfc-postmessage':
      if (!data.inputPort) {
        throw new Error("create_input: topology.input='rvfc-postmessage' but InitData.inputPort missing")
      }
      return createPostMessageInput(data.inputPort)
  }
}
