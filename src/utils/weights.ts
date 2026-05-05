import type { DataView_ } from '~/model/backend.ts'

// Coerce a weight-blob leaf (as stored on Conv2DWeights / DepthwiseWeights) to
// a typed view ready for backend.upload(). Float32Array stays as-is; Uint16Array
// stays as-is (raw fp16 bits from a .f16.bin loader); plain number[] (e.g. test
// fixtures parsed from JSON) is wrapped without copying values to fp16 — the
// backend handles f32→f16 conversion if needed.
export function toUploadView(data: ArrayLike<number>): DataView_ {
  if (data instanceof Float32Array || data instanceof Uint16Array) return data
  return new Float32Array(data)
}
