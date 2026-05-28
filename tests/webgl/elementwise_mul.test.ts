import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/elementwise_mul.json'

const THRESHOLD = 1e-4

interface MulFixture {
  channels: number
  input_shape: [number, number, number, number]
  input1: number[]
  input2: number[]
  expected_output: number[]
}

describe('ElementwiseMul (WebGL)', () => {
  it('matches PyTorch', async () => {
    const fx = fixture as MulFixture
    const backend = createWebGLBackend()

    const [, C, H, W] = fx.input_shape
    const a = backend.tensor(H, W, C, new Float32Array(fx.input1))
    const b = backend.tensor(H, W, C, new Float32Array(fx.input2))

    const op = backend.ops.ElementwiseMul(a, b)
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fx.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
