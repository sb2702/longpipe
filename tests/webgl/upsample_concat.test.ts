import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/upsample_concat.json'

const THRESHOLD = 1e-4

describe('UpsampleConcat (WebGL)', () => {
  it('upsample(a) + concat(b) matches PyTorch', async () => {
    const backend = createWebGLBackend()

    const a = backend.tensor(fixture.in_h,  fixture.in_w,  fixture.a_channels, new Float32Array(fixture.input_a))
    const b = backend.tensor(fixture.out_h, fixture.out_w, fixture.b_channels, new Float32Array(fixture.input_b))

    const op = backend.ops.UpsampleConcat(a, b, { outH: fixture.out_h, outW: fixture.out_w })
    op.run()

    const result = await backend.readback(op.output as WebGLTensor)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
