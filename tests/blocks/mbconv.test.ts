import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { MBConv } from '~/model/blocks/mbconv'

import mbconv_s1 from '../fixtures/mbconv_k3_s1_residual.json'
import mbconv_s2 from '../fixtures/mbconv_k3_s2.json'

const THRESHOLD = 1e-4

interface MBConvFixture {
  in_channels:  number
  mid_channels: number
  out_channels: number
  kernel_size:  number
  stride:       number
  padding:      number
  input_shape:  [number, number, number, number]
  input:           number[]
  expand_weights:  number[]
  expand_bias:     number[]
  dw_weights:      number[]
  dw_bias:         number[]
  proj_weights:    number[]
  proj_bias:       number[]
  expected_output: number[]
}

import { BACKENDS } from '../helpers/backends'

async function runFixture(backend: Backend, fixture: MBConvFixture) {
  const [, C, H, W] = fixture.input_shape
  const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

  const block = new MBConv(backend, input, {
    expand: { weights: fixture.expand_weights, bias: fixture.expand_bias },
    dw:     { weights: fixture.dw_weights,     bias: fixture.dw_bias },
    proj:   { weights: fixture.proj_weights,   bias: fixture.proj_bias },
  }, {
    inChannels:  fixture.in_channels,
    midChannels: fixture.mid_channels,
    outChannels: fixture.out_channels,
    kernel:      fixture.kernel_size,
    stride:      fixture.stride,
    padding:     fixture.padding,
  })

  block.run()

  const result = await backend.readback(block.output)

  const ref = new Float32Array(fixture.expected_output)
  let maxErr = 0
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
  return maxErr
}

describe.each(BACKENDS)('MBConv ($name)', ({ create }) => {
  it('k3 stride-1 with residual matches PyTorch', async () => {
    const backend = await create()
    expect(await runFixture(backend, mbconv_s1 as MBConvFixture)).toBeLessThan(THRESHOLD)
    backend.destroy()
  })

  it('k3 stride-2 no residual matches PyTorch', async () => {
    const backend = await create()
    expect(await runFixture(backend, mbconv_s2 as MBConvFixture)).toBeLessThan(THRESHOLD)
    backend.destroy()
  })
})
