import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/gru_fused.json'

const THRESHOLD = 1e-4

describe('CandUpdateFused (WebGL)', () => {
  it('candidate + update + output match PyTorch', async () => {
    const backend = createWebGLBackend()
    const { height: H, width: W } = fixture

    const uIn      = backend.tensor(H, W, 4, new Float32Array(fixture.u_in))
    const hPrev    = backend.tensor(H, W, 4, new Float32Array(fixture.h_prev))
    const gatesOut = backend.tensor(H, W, 4, new Float32Array(fixture.expected_gates))

    const op = backend.ops.CandUpdateFused(uIn, hPrev, gatesOut,
      { weights: fixture.cand_weights, bias: fixture.cand_bias }, fixture.gamma)
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
