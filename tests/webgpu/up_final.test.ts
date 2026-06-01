import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import upFinal from '../fixtures/up_final.json'
import upFinalSkip from '../fixtures/up_final_skip.json'

const THRESHOLD = 1e-4

function maxErr(result: Float32Array, expected: number[]) {
  const ref = new Float32Array(expected)
  let m = 0
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(result[i] - ref[i]))
  return m
}

describe('UpFinal (5→1 alpha head)', () => {
  it('concat(u, rgb) + 5→1 conv + sigmoid matches PyTorch', async () => {
    const backend = await createWebGPUBackend()
    const u   = backend.tensor(upFinal.in_h, upFinal.in_w, 4, new Float32Array(upFinal.u_in))
    const rgb = backend.tensor(upFinal.in_h, upFinal.in_w, 4, new Float32Array(upFinal.rgb_in))
    const op = backend.ops.UpFinal(u, rgb, { weights: upFinal.weights, bias: upFinal.bias })
    op.run()
    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()
    expect(maxErr(result, upFinal.expected_output)).toBeLessThan(THRESHOLD)
  })
})

describe('UpFinalSkip (9→1 alpha head)', () => {
  it('concat(u, d_full, rgb) + 9→1 conv + sigmoid matches PyTorch', async () => {
    const backend = await createWebGPUBackend()
    const u     = backend.tensor(upFinalSkip.in_h, upFinalSkip.in_w, 4, new Float32Array(upFinalSkip.u_in))
    const dFull = backend.tensor(upFinalSkip.in_h, upFinalSkip.in_w, 4, new Float32Array(upFinalSkip.d_in))
    const rgb   = backend.tensor(upFinalSkip.in_h, upFinalSkip.in_w, 4, new Float32Array(upFinalSkip.rgb_in))
    const op = backend.ops.UpFinalSkip(u, dFull, rgb, { weights: upFinalSkip.weights, bias: upFinalSkip.bias })
    op.run()
    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()
    expect(maxErr(result, upFinalSkip.expected_output)).toBeLessThan(THRESHOLD)
  })
})
