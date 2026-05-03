import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { MBConv } from '~/model/blocks/mbconv'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import mbconv_s1 from './fixtures/mbconv_k3_s1_residual.json'
import mbconv_s2 from './fixtures/mbconv_k3_s2.json'

const THRESHOLD = 1e-4

interface MBConvFixture {
  in_channels:  number
  mid_channels: number
  out_channels: number
  kernel_size:  number
  stride:       number
  padding:      number
  expand_ratio: number
  input_shape:  [number, number, number, number]  // NCHW
  input:           number[]
  expand_weights:  number[]
  expand_bias:     number[]
  dw_weights:      number[]
  dw_bias:         number[]
  proj_weights:    number[]
  proj_bias:       number[]
  expected_output: number[]
}

async function runFixture(fixture: MBConvFixture) {
  const backend = await WebGPUBackend.create()

  const [, C, H, W] = fixture.input_shape
  const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

  const block = new MBConv(backend, input, {
    expandWeights: backend.upload(new Float32Array(fixture.expand_weights)),
    expandBias:    backend.upload(new Float32Array(fixture.expand_bias)),
    dwWeights:     backend.upload(new Float32Array(fixture.dw_weights)),
    dwBias:        backend.upload(new Float32Array(fixture.dw_bias)),
    projWeights:   backend.upload(new Float32Array(fixture.proj_weights)),
    projBias:      backend.upload(new Float32Array(fixture.proj_bias)),
  }, {
    inChannels:  fixture.in_channels,
    midChannels: fixture.mid_channels,
    outChannels: fixture.out_channels,
    kernel:      fixture.kernel_size,
    stride:      fixture.stride,
    padding:     fixture.padding,
  })

  block.run()

  const result = await backend.readback(block.output as WebGPUTensor)
  backend.destroy()

  const ref = new Float32Array(fixture.expected_output)
  let maxErr = 0
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
  return maxErr
}

describe('MBConv', () => {
  it('k3 stride-1 with residual matches PyTorch', async () => {
    expect(await runFixture(mbconv_s1 as MBConvFixture)).toBeLessThan(THRESHOLD)
  })

  it('k3 stride-2 no residual matches PyTorch', async () => {
    expect(await runFixture(mbconv_s2 as MBConvFixture)).toBeLessThan(THRESHOLD)
  })
})
