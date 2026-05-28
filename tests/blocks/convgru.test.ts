import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { ConvGRU } from '~/model/blocks/convgru'

import fixture from '../fixtures/convgru_block.json'

const THRESHOLD = 1e-4

interface ConvGRUFixture {
  channels:    number
  passthrough: number
  recurrent:   number
  height:      number
  width:       number
  input_shape: [number, number, number, number]
  a:      number[]
  b:      number[]
  h_prev: number[]
  z_conv: { weights: number[]; bias: number[] }
  r_conv: { weights: number[]; bias: number[] }
  cand:   { weights: number[]; bias: number[] }
  gamma:  number[]
  expected_output: number[]
}

import { BACKENDS } from '../helpers/backends'

describe.each(BACKENDS)('ConvGRU ($name)', ({ create }) => {
  it('single-timestep forward matches PyTorch', async () => {
    const fx = fixture as ConvGRUFixture
    const backend = await create()

    const { passthrough, recurrent: c, height: H, width: W } = fx
    const a      = passthrough > 0
      ? backend.tensor(H, W, passthrough, new Float32Array(fx.a))
      : null
    const b      = backend.tensor(H, W, c, new Float32Array(fx.b))
    const h_prev = backend.tensor(H, W, c, new Float32Array(fx.h_prev))

    const block = new ConvGRU(backend, a, b, h_prev, {
      zConv: fx.z_conv,
      rConv: fx.r_conv,
      cand:  fx.cand,
      gamma: fx.gamma,
    }, {
      passthrough,
      recurrent: c,
    })
    block.run()

    const result = await backend.readback(block.output)
    backend.destroy()

    const ref = new Float32Array(fx.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
