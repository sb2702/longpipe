import type { ModelWeights } from '~/model/weights'
import type { Dtype } from '~/model/backend'

// Binary pack format (see sdk/WEIGHTS_FORMAT.md):
//   [0..4)        uint32 LE: header JSON length
//   [4..4+N)      utf-8 JSON: ModelWeights shape, with every leaf number[]
//                 replaced by { offset, length } in element count, plus an
//                 optional top-level "__dtype__": "f32" | "f16" tag (defaults
//                 to "f32" for back-compat with pre-fp16 .bin files)
//   [4+N..)       packed array payload, in offset order; element width is
//                 4 bytes for f32, 2 bytes for f16
//
// Leaves become typed views over the original ArrayBuffer — no copy. f32 files
// produce Float32Array views; f16 files produce Uint16Array views holding raw
// IEEE 754 binary16 bits, which the backend decodes / re-encodes as needed.

interface ArrayRef { offset: number; length: number }

const isArrayRef = (v: unknown): v is ArrayRef =>
  typeof v === 'object' && v !== null
  && typeof (v as ArrayRef).offset === 'number'
  && typeof (v as ArrayRef).length === 'number'

export function loadWeightsFromBinary(buf: ArrayBuffer): ModelWeights {
  if (buf.byteLength < 4) throw new Error('weights buffer too small')

  const headerLen = new DataView(buf, 0, 4).getUint32(0, true)
  const headerEnd = 4 + headerLen
  if (buf.byteLength < headerEnd) throw new Error('weights buffer truncated (header)')

  const headerBytes = new Uint8Array(buf, 4, headerLen)
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, unknown>

  const dtype: Dtype = (header['__dtype__'] as Dtype | undefined) ?? 'f32'
  const elemBytes = dtype === 'f16' ? 2 : 4

  // Float32Array offset must be 4-byte aligned; Uint16Array offset must be
  // 2-byte aligned. Python writer always pads to 4 bytes, so this satisfies
  // both — keep the strict check anyway.
  if (headerEnd % elemBytes !== 0)
    throw new Error(`payload not ${elemBytes}-byte aligned (header=${headerLen})`)

  const payloadElems = (buf.byteLength - headerEnd) / elemBytes

  const resolve = (node: unknown): unknown => {
    if (isArrayRef(node)) {
      const { offset, length } = node
      if (offset + length > payloadElems)
        throw new Error(`array ref out of range: offset=${offset} length=${length}`)
      const byteOffset = headerEnd + offset * elemBytes
      return dtype === 'f16'
        ? new Uint16Array(buf, byteOffset, length)
        : new Float32Array(buf, byteOffset, length)
    }
    if (Array.isArray(node)) return node.map(resolve)
    if (typeof node === 'object' && node !== null) {
      const out: Record<string, unknown> = {}
      for (const k of Object.keys(node)) {
        if (k === '__dtype__') continue
        out[k] = resolve((node as Record<string, unknown>)[k])
      }
      return out
    }
    return node
  }

  return resolve(header) as ModelWeights
}
