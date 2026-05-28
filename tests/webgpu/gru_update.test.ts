import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import { GruUpdateWebGPU } from '~/model/backends/webgpu/ops/gru_update'
import type { WebGPUTensor } from '~/model/backends/webgpu/base_webgpu_op'

import fixture from '../fixtures/gru_update.json'

const THRESHOLD = 1e-4

interface GruFixture {
  channels: number
  input_shape: [number, number, number, number]
  z:      number[]
  h_prev: number[]
  h_til:  number[]
  expected_output: number[]
}

describe('GruUpdate (WebGPU)', () => {
  it('fused (1-z)*h_prev + z*h_til matches PyTorch', async () => {
    const fx = fixture as GruFixture
    const backend = await createWebGPUBackend()

    const [, C, H, W] = fx.input_shape
    const z      = backend.tensor(H, W, C, new Float32Array(fx.z))
    const h_prev = backend.tensor(H, W, C, new Float32Array(fx.h_prev))
    const h_til  = backend.tensor(H, W, C, new Float32Array(fx.h_til))

    const op = new GruUpdateWebGPU(backend, z, h_prev, h_til)
    op.run()

    const result = await backend.readback(op.output as WebGPUTensor)
    backend.destroy()

    const ref = new Float32Array(fx.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
