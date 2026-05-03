import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { AddWebGPU } from '~/model/backends/webgpu/ops/add'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import elementwise_add from '../fixtures/elementwise_add.json'

const THRESHOLD = 1e-4

interface AddFixture {
  channels: number
  input_shape: [number, number, number, number]  // NCHW
  input1: number[]
  input2: number[]
  expected_output: number[]
}

describe('Add', () => {
  it('elementwise add matches PyTorch', async () => {
    const fixture = elementwise_add as AddFixture
    const backend = await WebGPUBackend.create()

    const [, C, H, W] = fixture.input_shape
    const a = backend.tensor(H, W, C, new Float32Array(fixture.input1))
    const b = backend.tensor(H, W, C, new Float32Array(fixture.input2))

    const op = new AddWebGPU(backend, a, b)
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
