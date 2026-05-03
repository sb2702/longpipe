import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/upsample_conv1x1.json'

const THRESHOLD = 1e-4

describe('UpsampleConv1x1 (WebGPU)', () => {
  it('upsample + 1x1 conv matches PyTorch', async () => {
    const backend = await WebGPUBackend.create()

    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

    const op = backend.ops.UpsampleConv1x1(input, { weights: fixture.weights, bias: fixture.bias }, {
      outH:        fixture.out_h,
      outW:        fixture.out_w,
      outChannels: fixture.out_channels,
      activation:  'none',
    })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
