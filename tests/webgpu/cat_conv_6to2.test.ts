import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/cat_conv_6to2.json'

const THRESHOLD = 1e-4

describe('CatConv6to2', () => {
  it('concat(u,d) + 6→2 conv 3x3 + relu matches PyTorch', async () => {
    const backend = await createWebGPUBackend()

    const u = backend.tensor(fixture.in_h, fixture.in_w, 4, new Float32Array(fixture.u_in))
    const d = backend.tensor(fixture.in_h, fixture.in_w, 4, new Float32Array(fixture.d_in))

    const op = backend.ops.CatConv6to2(u, d, { weights: fixture.weights, bias: fixture.bias })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
