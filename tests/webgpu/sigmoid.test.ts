import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import { SigmoidWebGPU } from '~/model/backends/webgpu/ops/sigmoid'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/sigmoid.json'

const THRESHOLD = 1e-4

describe('Sigmoid (WebGPU)', () => {
  it('matches PyTorch', async () => {
    const backend = await createWebGPUBackend()

    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

    const op = new SigmoidWebGPU(backend, input)
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
