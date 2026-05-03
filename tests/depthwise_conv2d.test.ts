import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { DepthwiseConv2DWebGPU } from '~/model/backends/webgpu/ops/depthwise_conv2d'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import depthwise_3x3 from './fixtures/depthwise_3x3.json'

const THRESHOLD = 1e-4

interface DepthwiseFixture {
  kernel_size: number
  stride: number
  padding: number
  channels: number
  input_shape: [number, number, number, number]  // NCHW
  input: number[]
  weights: number[]
  bias: number[]
  expected_output: number[]
}

describe('DepthwiseConv2d', () => {
  it('3x3 matches PyTorch', async () => {
    const fixture = depthwise_3x3 as DepthwiseFixture
    const backend = await WebGPUBackend.create()

    const [, C, H, W] = fixture.input_shape
    const input   = backend.tensor(H, W, C, new Float32Array(fixture.input))
    const weights = backend.upload(new Float32Array(fixture.weights))
    const bias    = backend.upload(new Float32Array(fixture.bias))

    const op = new DepthwiseConv2DWebGPU(backend, input, weights, bias, {
      kernel:     fixture.kernel_size,
      stride:     fixture.stride,
      padding:    fixture.padding,
      activation: 'none',
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
