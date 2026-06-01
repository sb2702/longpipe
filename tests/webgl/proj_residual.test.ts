import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/proj_residual.json'

const THRESHOLD = 1e-4

describe('ProjResidual (WebGL)', () => {
  it('1x1 proj + residual matches PyTorch', async () => {
    const backend = createWebGLBackend()

    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))
    const skip  = backend.tensor(H, W, fixture.out_channels, new Float32Array(fixture.skip))

    const op = backend.ops.ProjResidual(input, skip,
      { weights: fixture.weights, bias: fixture.bias },
      { outChannels: fixture.out_channels })
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
