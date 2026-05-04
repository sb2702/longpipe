import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import { ChannelConcatWebGPU } from '~/model/backends/webgpu/ops/channel_concat'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/channel_concat.json'

const THRESHOLD = 1e-4

describe('ChannelConcat (WebGPU)', () => {
  it('cat(a, b, dim=channel) matches PyTorch', async () => {
    const backend = await createWebGPUBackend()

    const [, , H, W] = fixture.input_shape
    const a = backend.tensor(H, W, fixture.a_channels, new Float32Array(fixture.input_a))
    const b = backend.tensor(H, W, fixture.b_channels, new Float32Array(fixture.input_b))

    const op = new ChannelConcatWebGPU(backend, a, b)
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
