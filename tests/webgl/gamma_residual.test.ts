import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/gamma_residual.json'

const THRESHOLD = 1e-4

interface GammaFixture {
  channels: number
  height: number
  width:  number
  input_shape: [number, number, number, number]
  b:     number[]
  h_new: number[]
  gamma: number[]
  expected_output: number[]
}

describe('GammaResidual (WebGL)', () => {
  it('b + γ ⊙ h_new matches PyTorch', async () => {
    const fx = fixture as GammaFixture
    const backend = createWebGLBackend()

    const [, C, H, W] = fx.input_shape
    const b     = backend.tensor(H, W, C, new Float32Array(fx.b))
    const h_new = backend.tensor(H, W, C, new Float32Array(fx.h_new))

    const op = backend.ops.GammaResidual(b, h_new, new Float32Array(fx.gamma))
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fx.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
