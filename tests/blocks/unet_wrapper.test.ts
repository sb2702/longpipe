import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { UNetWrapper, type UNetVariant } from '~/model/blocks/unet_wrapper'

import wrapperA from '../fixtures/wrapper_a.json'
import wrapperB from '../fixtures/wrapper_b.json'
import wrapperE from '../fixtures/wrapper_e.json'
import wrapperD from '../fixtures/wrapper_d.json'

const THRESHOLD = 1e-4

interface WrapperFixture {
  variant: UNetVariant
  c_high:  number
  c_low:   number
  c_up:    number
  feat_ch: number
  base_hw: number
  canvas:  number
  x_hr:    number[]
  feat_lr: number[]
  down1:      { weights: number[]; bias: number[] }
  down2?:     { weights: number[]; bias: number[] }
  adapter:    { weights: number[]; bias: number[] }
  expandFeat: { weights: number[]; bias: number[] }
  up1Combine?: { weights: number[]; bias: number[] }
  upCombine:  { weights: number[]; bias: number[] }
  expected_output: number[]
}

function extractChannel0(vec4Buf: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count)
  for (let i = 0; i < count; i++) out[i] = vec4Buf[i * 4]
  return out
}

import { BACKENDS } from '../helpers/backends'

async function runWrapperFixture(backend: Backend, fx: WrapperFixture) {
  const { canvas, base_hw, feat_ch } = fx

  // x_hr is 3-channel RGB padded to 4 channels at fixture-gen time.
  const x_hr    = backend.tensor(canvas, canvas, 4,        new Float32Array(fx.x_hr))
  const feat_lr = backend.tensor(base_hw, base_hw, feat_ch, new Float32Array(fx.feat_lr))

  const block = new UNetWrapper(backend, x_hr, feat_lr, {
    down1:      fx.down1,
    down2:      fx.down2,
    adapter:    fx.adapter,
    expandFeat: fx.expandFeat,
    up1Combine: fx.up1Combine,
    upCombine:  fx.upCombine,
  }, {
    variant: fx.variant,
    cHigh:   fx.c_high,
    cLow:    fx.c_low,
    cUp:     fx.c_up,
  })
  block.run()

  // Sigmoid output is 4-channel padded; first channel is the real alpha and
  // channels 1..3 are sigmoid(0)=0.5 from the padded logits. Extract ch 0.
  const sigOut = await backend.readback(block.output)
  return extractChannel0(sigOut, canvas * canvas)
}

describe.each(BACKENDS)('UNetWrapper A ($name)', ({ create }) => {
  it('one-stage stride-2 down, end-to-end matches PyTorch', async () => {
    const backend = await create()
    const result = await runWrapperFixture(backend, wrapperA as WrapperFixture)
    backend.destroy()

    const ref = new Float32Array((wrapperA as WrapperFixture).expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})

describe.each(BACKENDS)('UNetWrapper B ($name)', ({ create }) => {
  it('single stride-3 down, end-to-end matches PyTorch', async () => {
    const backend = await create()
    const result = await runWrapperFixture(backend, wrapperB as WrapperFixture)
    backend.destroy()

    const ref = new Float32Array((wrapperB as WrapperFixture).expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})

describe.each(BACKENDS)('UNetWrapper E ($name)', ({ create }) => {
  it('two stride-2 down, end-to-end matches PyTorch', async () => {
    const backend = await create()
    const result = await runWrapperFixture(backend, wrapperE as WrapperFixture)
    backend.destroy()

    const ref = new Float32Array((wrapperE as WrapperFixture).expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})

describe.each(BACKENDS)('UNetWrapper D ($name)', ({ create }) => {
  it('fractional first stage + full-res skip (9→1 head) matches PyTorch', async () => {
    const backend = await create()
    const result = await runWrapperFixture(backend, wrapperD as WrapperFixture)
    backend.destroy()

    const ref = new Float32Array((wrapperD as WrapperFixture).expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
