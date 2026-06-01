import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/gru_fused.json'

const THRESHOLD = 1e-4

describe('GatesFused', () => {
  it('z + r gates match PyTorch', async () => {
    const backend = await createWebGPUBackend()
    const { height: H, width: W } = fixture

    const uIn   = backend.tensor(H, W, 4, new Float32Array(fixture.u_in))
    const hPrev = backend.tensor(H, W, 4, new Float32Array(fixture.h_prev))

    const op = backend.ops.GatesFused(uIn, hPrev,
      { weights: fixture.gates_weights, bias: fixture.gates_bias })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_gates)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
