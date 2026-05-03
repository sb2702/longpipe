import { describe, it, expect } from 'vitest'
import { WebGLBackend } from '~/model/backends/webgl/index'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/channel_concat.json'

const THRESHOLD = 1e-4

describe('ChannelConcat (WebGL)', () => {
  it('cat(a, b, dim=channel) matches PyTorch', async () => {
    const backend = WebGLBackend.create()

    const [, , H, W] = fixture.input_shape
    const a = backend.tensor(H, W, fixture.a_channels, new Float32Array(fixture.input_a))
    const b = backend.tensor(H, W, fixture.b_channels, new Float32Array(fixture.input_b))

    const op = backend.ops.ChannelConcat(a, b)
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
