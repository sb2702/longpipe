import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/conv2d_add.json'

const THRESHOLD = 1e-4

describe('Conv2dAdd (WebGPU)', () => {
  it('conv + skip add matches PyTorch', async () => {
    const backend = await WebGPUBackend.create()

    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))
    const skip  = backend.tensor(H, W, fixture.out_channels, new Float32Array(fixture.skip))

    const op = backend.ops.Conv2dAdd(input, skip, { weights: fixture.weights, bias: fixture.bias }, {
      outChannels: fixture.out_channels,
      kernel:      fixture.kernel_size,
      stride:      fixture.stride,
      padding:     fixture.padding,
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
