import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/gru_fused.json'

const THRESHOLD = 1e-4

describe('GatesFused (WebGL)', () => {
  it('z + r gates match PyTorch', async () => {
    const backend = createWebGLBackend()
    const { height: H, width: W } = fixture

    const uIn   = backend.tensor(H, W, 4, new Float32Array(fixture.u_in))
    const hPrev = backend.tensor(H, W, 4, new Float32Array(fixture.h_prev))

    const op = backend.ops.GatesFused(uIn, hPrev,
      { weights: fixture.gates_weights, bias: fixture.gates_bias })
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_gates)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
