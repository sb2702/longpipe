import { describe, it, expect } from 'vitest'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import type { ModelWeights } from '~/model/weights'

import jsonFixture from '../fixtures/model_large.json'
import binUrl      from '../../weights/model_large.bin?url'

// Walk two ModelWeights in lockstep and collect (path, jsonArray, binArray)
// pairs at every leaf weights/bias array.
type Pair = { path: string; a: ArrayLike<number>; b: ArrayLike<number> }

function collectPairs(json: unknown, bin: unknown, path = ''): Pair[] {
  const isLeafNumberArray = (v: unknown): v is number[] =>
    Array.isArray(v) && (v.length === 0 || typeof v[0] === 'number')

  if (isLeafNumberArray(json)) {
    if (!(bin instanceof Float32Array))
      throw new Error(`expected Float32Array at ${path}, got ${typeof bin}`)
    return [{ path, a: json, b: bin }]
  }

  if (Array.isArray(json)) {
    if (!Array.isArray(bin)) throw new Error(`shape mismatch at ${path}: array vs non-array`)
    expect(bin.length, `array length at ${path}`).toBe(json.length)
    return json.flatMap((item, i) => collectPairs(item, bin[i], `${path}[${i}]`))
  }

  if (typeof json === 'object' && json !== null) {
    if (typeof bin !== 'object' || bin === null)
      throw new Error(`shape mismatch at ${path}: object vs non-object`)
    const jKeys = Object.keys(json).sort()
    const bKeys = Object.keys(bin).sort()
    expect(bKeys, `keys at ${path}`).toEqual(jKeys)
    return jKeys.flatMap(k =>
      collectPairs(
        (json as Record<string, unknown>)[k],
        (bin  as Record<string, unknown>)[k],
        path ? `${path}.${k}` : k,
      ),
    )
  }

  return []
}

describe('loadWeightsFromBinary', () => {
  it('produces ModelWeights identical to the JSON fixture (large preset)', async () => {
    const ab = await fetch(binUrl).then(r => r.arrayBuffer())

    const binWeights  = loadWeightsFromBinary(ab)
    const jsonWeights = (jsonFixture as { weights: ModelWeights }).weights

    const pairs = collectPairs(jsonWeights, binWeights)
    expect(pairs.length, 'leaf array count').toBeGreaterThan(0)

    let totalFloats = 0
    for (const { path, a, b } of pairs) {
      expect(b.length, `length at ${path}`).toBe(a.length)
      for (let i = 0; i < a.length; i++) {
        // JSON values were originally float32, JSON-encoded, parsed back as
        // doubles. Casting through Float32Array round-trips them exactly.
        const expected = Math.fround(a[i])
        if (b[i] !== expected) {
          throw new Error(
            `mismatch at ${path}[${i}]: json=${expected} bin=${b[i]}`,
          )
        }
      }
      totalFloats += a.length
    }

    expect(totalFloats).toBeGreaterThan(3_000_000)  // ~3.21M for large
  })
})
