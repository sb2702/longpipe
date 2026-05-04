import type { ModelWeights } from '~/model/weights'

// Binary pack format (see sdk/WEIGHTS_FORMAT.md):
//   [0..4)        uint32 LE: header JSON length
//   [4..4+N)      utf-8 JSON: ModelWeights shape with every leaf number[]
//                 replaced by { offset, length } in float32 elements
//   [4+N..)       packed float32 arrays, in offset order
//
// Leaves (Conv2DWeights / DepthwiseWeights `weights` and `bias` fields) become
// Float32Array views over the original ArrayBuffer — no copy.

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
  const header = JSON.parse(new TextDecoder().decode(headerBytes))

  // float32 arrays start immediately after the header, aligned to 4 bytes.
  if (headerEnd % 4 !== 0)
    throw new Error(`payload not 4-byte aligned (header=${headerLen})`)

  const payloadFloats = (buf.byteLength - headerEnd) / 4

  const resolve = (node: unknown): unknown => {
    if (isArrayRef(node)) {
      const { offset, length } = node
      if (offset + length > payloadFloats)
        throw new Error(`array ref out of range: offset=${offset} length=${length}`)
      return new Float32Array(buf, headerEnd + offset * 4, length)
    }
    if (Array.isArray(node)) return node.map(resolve)
    if (typeof node === 'object' && node !== null) {
      const out: Record<string, unknown> = {}
      for (const k of Object.keys(node)) out[k] = resolve((node as Record<string, unknown>)[k])
      return out
    }
    return node
  }

  return resolve(header) as ModelWeights
}
