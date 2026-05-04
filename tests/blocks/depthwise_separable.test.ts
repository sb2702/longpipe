import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { DepthwiseSeparable } from '~/model/blocks/depthwise_separable'

import fixture from '../fixtures/depthwise_separable.json'

const THRESHOLD = 1e-4

import { BACKENDS } from '../helpers/backends'

describe.each(BACKENDS)('DepthwiseSeparable ($name)', ({ create }) => {
  it('dw+relu6+pw matches PyTorch', async () => {
    const backend = await create()

    const [, C, H, W] = fixture.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

    const block = new DepthwiseSeparable(backend, input, { dw: fixture.dw, pw: fixture.pw }, {
      outChannels: fixture.out_channels,
      kernel:      fixture.kernel_size,
      stride:      fixture.stride,
      padding:     fixture.padding,
    })
    block.run()

    const result = await backend.readback(block.output)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
