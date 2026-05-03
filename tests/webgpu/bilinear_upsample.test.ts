import { describe, it, expect } from 'vitest'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { BilinearUpsampleWebGPU } from '~/model/backends/webgpu/ops/bilinear_upsample'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/bilinear_upsample_2x.json'

const THRESHOLD = 1e-4

describe('BilinearUpsample (WebGPU)', () => {
  it('2× matches PyTorch align_corners=False', async () => {
    const backend = await WebGPUBackend.create()

    const [, C, H, W] = fixture.input_shape
    const [,, outH, outW] = fixture.output_shape
    const input = backend.tensor(H, W, C, new Float32Array(fixture.input))

    const op = new BilinearUpsampleWebGPU(backend, input, { outH, outW })
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
