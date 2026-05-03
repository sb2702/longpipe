import { describe, it, expect } from 'vitest'
import { WebGLBackend } from '~/model/backends/webgl/index'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import conv2d_1x1    from '../fixtures/conv2d_1x1.json'
import conv2d_3x3    from '../fixtures/conv2d_3x3.json'
import conv2d_3x3_s2 from '../fixtures/conv2d_3x3_stride2.json'

const THRESHOLD = 1e-4

interface Conv2dFixture {
  kernel_size:  number
  stride:       number
  padding:      number
  in_channels:  number
  out_channels: number
  input_shape:  [number, number, number, number]
  input:           number[]
  weights:         number[]
  bias:            number[]
  expected_output: number[]
}

async function runFixture(fixture: Conv2dFixture) {
  const backend = WebGLBackend.create()

  const [, C, H, W] = fixture.input_shape
  const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

  const op = backend.ops.Conv2d(input, { weights: fixture.weights, bias: fixture.bias }, {
    outChannels: fixture.out_channels,
    kernel:      fixture.kernel_size,
    stride:      fixture.stride,
    padding:     fixture.padding,
    activation:  'none',
  })
  op.run()

  const result = await backend.readback(op.output as WebGLTensor)
  backend.destroy()

  const ref = new Float32Array(fixture.expected_output)
  let maxErr = 0
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
  return maxErr
}

describe('Conv2d (WebGL)', () => {
  it('1x1 matches PyTorch',              async () => expect(await runFixture(conv2d_1x1    as Conv2dFixture)).toBeLessThan(THRESHOLD))
  it('3x3 same padding matches PyTorch', async () => expect(await runFixture(conv2d_3x3    as Conv2dFixture)).toBeLessThan(THRESHOLD))
  it('3x3 stride-2 matches PyTorch',     async () => expect(await runFixture(conv2d_3x3_s2 as Conv2dFixture)).toBeLessThan(THRESHOLD))
})
