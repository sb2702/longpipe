import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { floatToHalf, halfToFloat, float32ArrayToHalf } from '~/utils/fp16'
import { createWebGPUBackend, createWebGLBackend } from '../helpers/backends'

import conv2d_3x3 from '../fixtures/conv2d_3x3.json'

interface Conv2dFixture {
  kernel_size:  number
  stride:       number
  padding:      number
  in_channels:  number
  out_channels: number
  input_shape:  [number, number, number, number]
  input:           number[]
  weights:         number[]
  bias:            number[]
  expected_output: number[]
}

// Build a minimal .f16.bin in memory: header tags __dtype__='f16'; payload is
// raw fp16 bits packed at 4-byte alignment after the length prefix + header.
function packF16Bin(weights: Float32Array, bias: Float32Array): ArrayBuffer {
  const header = JSON.stringify({
    __dtype__: 'f16',
    encoder: { stem: {
      weights: { offset: 0,             length: weights.length },
      bias:    { offset: weights.length, length: bias.length    },
    }},
  })
  let headerBytes = new TextEncoder().encode(header)
  // JS `%` returns negative remainders for negative dividends; the Python
  // writer uses the same shape `(-(4+N)) % 4` and gets a non-negative pad.
  // Mirror that here.
  const pad = (4 - ((4 + headerBytes.length) % 4)) % 4
  const padded = new Uint8Array(headerBytes.length + pad)
  padded.set(headerBytes)
  for (let i = headerBytes.length; i < padded.length; i++) padded[i] = 0x20 // space

  const wBits = float32ArrayToHalf(weights)
  const bBits = float32ArrayToHalf(bias)

  const payloadBytes = wBits.byteLength + bBits.byteLength
  const total = 4 + padded.length + payloadBytes
  const buf = new ArrayBuffer(total)
  new DataView(buf).setUint32(0, padded.length, true)
  new Uint8Array(buf, 4, padded.length).set(padded)
  new Uint16Array(buf, 4 + padded.length,                 wBits.length).set(wBits)
  new Uint16Array(buf, 4 + padded.length + wBits.byteLength, bBits.length).set(bBits)
  return buf
}

describe('fp16: utility round-trip', () => {
  it('floatToHalf / halfToFloat covers normals, subnormals, zero, inf, NaN', () => {
    const samples = [0, -0, 1, -1, 0.5, 65504, 6.10e-5, 1e-7, 1e-10]
    for (const v of samples) {
      const round = halfToFloat(floatToHalf(v))
      // fp16 relative precision ~1/2048; absolute floor ~6e-8 for subnormals.
      const tol = Math.max(Math.abs(v) * 1e-3, 1e-7)
      expect(Math.abs(round - v)).toBeLessThan(tol)
    }
    expect(halfToFloat(floatToHalf(Infinity))).toBe(Infinity)
    expect(halfToFloat(floatToHalf(-Infinity))).toBe(-Infinity)
    expect(Number.isNaN(halfToFloat(floatToHalf(NaN)))).toBe(true)
    expect(floatToHalf(1e10)).toBe(0x7c00) // overflow → +Inf bits
  })
})

describe('fp16: loadWeightsFromBinary recognizes __dtype__', () => {
  it('returns Uint16Array views for f16 file', () => {
    const w = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    const b = new Float32Array([0.5, -0.5, 1.5, -1.5])
    const buf = packF16Bin(w, b)

    const loaded = loadWeightsFromBinary(buf as never) as never as {
      encoder: { stem: { weights: Uint16Array; bias: Uint16Array } }
    }
    expect(loaded.encoder.stem.weights).toBeInstanceOf(Uint16Array)
    expect(loaded.encoder.stem.bias).toBeInstanceOf(Uint16Array)
    expect(loaded.encoder.stem.weights.length).toBe(16)
    expect(loaded.encoder.stem.bias.length).toBe(4)
    // Spot-check a couple of decoded values.
    expect(halfToFloat(loaded.encoder.stem.weights[0])).toBeCloseTo(1, 3)
    expect(halfToFloat(loaded.encoder.stem.bias[0])).toBeCloseTo(0.5, 3)
  })
})

// fp16 backend availability is asymmetric: WebGPU needs `shader-f16`, WebGL
// always supports RGBA16F + HALF_FLOAT (storage-only). Skip fp16 WebGPU if
// the device lacks the feature.
async function tryCreateWebGPU(): Promise<Backend | null> {
  try { return await createWebGPUBackend('f16') }
  catch (e) { console.warn('[fp16-smoke] WebGPU f16 unavailable:', (e as Error).message); return null }
}

const fp16Backends: Array<{ name: string; create: () => Promise<Backend | null> }> = [
  { name: 'WebGPU', create: () => tryCreateWebGPU() },
  { name: 'WebGL',  create: async () => createWebGLBackend('f16') },
]

const F16_THRESHOLD = 5e-2  // fp16 ~3-4 decimal digits; magnitudes up to ~10 in the fixture

describe.each(fp16Backends)('fp16: conv2d 3×3 ($name)', ({ create }) => {
  it('matches PyTorch within fp16 tolerance', async () => {
    const backend = await create()
    if (!backend) return  // skipped: fp16 unavailable on this backend

    const fixture = conv2d_3x3 as Conv2dFixture
    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

    const op = backend.ops.Conv2d(input, { weights: fixture.weights, bias: fixture.bias }, {
      outChannels: fixture.out_channels,
      kernel:      fixture.kernel_size,
      stride:      fixture.stride,
      padding:     fixture.padding,
      activation:  'none',
    })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(F16_THRESHOLD)
  })
})
