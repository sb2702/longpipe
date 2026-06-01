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

// Pad a short weight leaf (e.g. a 1- or 2-element fused-op bias/gamma) up to a
// whole vec4/RGBA texel so backends can upload it as one element. WebGPU needs
// a full vec4 of backing storage; WebGL needs exactly w*h*4 floats for a 1×1
// texture. Extra lanes are zero. Returns f32 (backends convert to f16 on upload).
export function padToVec4(data: ArrayLike<number>): Float32Array {
  const n = Math.max(4, Math.ceil(data.length / 4) * 4)
  const out = new Float32Array(n)
  for (let i = 0; i < data.length; i++) out[i] = data[i]
  return out
}
