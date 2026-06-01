import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import proj_residual from '../fixtures/proj_residual.json'

const THRESHOLD = 1e-4

interface ProjResidualFixture {
  in_channels:  number
  out_channels: number
  input_shape:  [number, number, number, number]
  input:           number[]
  skip:            number[]
  weights:         number[]
  bias:            number[]
  expected_output: number[]
}

async function runFixture(fixture: ProjResidualFixture) {
  const backend = await createWebGPUBackend()

  const [, C, H, W] = fixture.input_shape
  const input = backend.tensor(H, W, C, new Float32Array(fixture.input))
  const skip  = backend.tensor(H, W, fixture.out_channels, new Float32Array(fixture.skip))

  const op = backend.ops.ProjResidual(input, skip,
    { weights: fixture.weights, bias: fixture.bias },
    { outChannels: fixture.out_channels })
  op.run()

  const result = await backend.readback(op.output as WebGPUTensor)
  backend.destroy()

  const ref = new Float32Array(fixture.expected_output)
  let maxErr = 0
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
  return maxErr
}

describe('ProjResidual', () => {
  it('1x1 proj + residual matches PyTorch', async () =>
    expect(await runFixture(proj_residual as ProjResidualFixture)).toBeLessThan(THRESHOLD))
})
