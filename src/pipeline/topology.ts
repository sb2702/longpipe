// Per-browser transport selection. Two axes (input, output) chosen
// independently; the worker's renderer is identical across all 6 combos.
// See docs/PIPELINE.md for the empirical browser support matrix.

export type InputPath  = 'mstp' | 'rvfc-postmessage'
export type OutputPath = 'mstg' | 'transfer-capture' | 'bitmap-shuttle'

export interface Topology {
  input:  InputPath
  output: OutputPath
}

export function selectTopology(): Topology {
  return {
    input:  'MediaStreamTrackProcessor' in self ? 'mstp' : 'rvfc-postmessage',
    output: 'MediaStreamTrackGenerator' in self ? 'mstg'
          :  canTransferCanvasAndCapture()      ? 'transfer-capture'
          :                                       'bitmap-shuttle',
  }
}

// Probe whether `canvas.captureStream()` keeps emitting frames after
// `canvas.transferControlToOffscreen()`. Result cached per session.
// Firefox throws an explicit DOMException on the captureStream call when
// the canvas has been transferred.
let _canTransferCache: boolean | null = null

export function canTransferCanvasAndCapture(): boolean {
  if (_canTransferCache !== null) return _canTransferCache
  // TODO: real probe — create canvas, captureStream(), transferControlToOffscreen,
  // catch DOMException. For now use UA inference (safe lower bound).
  _canTransferCache = !/Firefox/.test(navigator.userAgent)
  return _canTransferCache
}
