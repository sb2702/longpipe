import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/concat_conv2d.json'

const THRESHOLD = 1e-4

describe('ConcatConv2d', () => {
  it('concat + 3x3 conv + relu6 matches PyTorch', async () => {
    const backend = await createWebGPUBackend()

    const a = backend.tensor(fixture.in_h, fixture.in_w, fixture.a_channels, new Float32Array(fixture.input_a))
    const b = backend.tensor(fixture.in_h, fixture.in_w, fixture.b_channels, new Float32Array(fixture.input_b))

    const op = backend.ops.ConcatConv2d(a, b,
      { weights: fixture.weights, bias: fixture.bias },
      { outChannels: fixture.out_channels })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
