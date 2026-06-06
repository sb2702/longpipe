import type { DataView_ } from '~/model/backend.ts'
import { halfToFloat } from '~/utils/fp16.ts'

// Coerce a weight-blob leaf (as stored on Conv2DWeights / DepthwiseWeights) to
// a typed view ready for backend.upload(). Float32Array stays as-is; Uint16Array
// stays as-is (raw fp16 bits from a .f16.bin loader); plain number[] (e.g. test
// fixtures parsed from JSON) is wrapped without copying values to fp16 — the
// backend handles f32→f16 conversion if needed.
export function toUploadView(data: ArrayLike<number>): DataView_ {
  if (data instanceof Float32Array || data instanceof Uint16Array) return data
  return new Float32Array(data)
}

// Pad a short weight leaf (e.g. a 1- or 2-element fused-op bias/gamma) up to a
// whole vec4/RGBA texel so backends can upload it as one element. WebGPU needs
// a full vec4 of backing storage; WebGL needs exactly w*h*4 floats for a 1×1
// texture. Extra lanes are zero. Returns f32 (backends convert to f16 on upload).
//
// A Uint16Array leaf is raw fp16 BITS (from a .f16.bin loader), so it must be
// decoded to float values here — `out[i] = data[i]` would otherwise copy the
// integer bit pattern (e.g. 0x3C00 = 15360 instead of 1.0), which the backend
// then f32→f16-converts into garbage. (Unlike toUploadView, this reads VALUES.)
export function padToVec4(data: ArrayLike<number>): Float32Array {
  const n = Math.max(4, Math.ceil(data.length / 4) * 4)
  const out = new Float32Array(n)
  const half = data instanceof Uint16Array
  for (let i = 0; i < data.length; i++) out[i] = half ? halfToFloat(data[i]) : data[i]
  return out
}
