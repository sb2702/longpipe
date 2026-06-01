import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import type { ModelWeights, UNetWrapperWeights } from '~/model/weights'
import { TierModel, type BaseNetworkCtor } from '~/model/tier_model'
import type { UNetVariant } from '~/model/blocks/unet_wrapper'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small'
import { EfficientNetLiteMattingLarge } from '~/model/networks/efficientnetlite_matting_large'
import { EfficientNetLiteMattingXL }    from '~/model/networks/efficientnetlite_matting_xl'
import { BACKENDS } from '../helpers/backends'

import tierXs     from '../fixtures/tier_full_xs.json'
import tierSmall  from '../fixtures/tier_full_small.json'
import tierMedium from '../fixtures/tier_full_medium.json'
import tierLarge  from '../fixtures/tier_full_large.json'
import tierXl     from '../fixtures/tier_full_xl.json'

// Full base+wrapper composition (down → base → up). Each op/block/base/wrapper
// is already validated layer-by-layer elsewhere; this checks the WIRING with
// end-to-end alpha vs the PyTorch base+wrapper forward.
const THRESHOLD = 1e-3

interface TierFullFixture {
  name:    string
  variant: string
  c_high:  number
  c_low:   number
  c_up:    number
  canvas_h: number
  canvas_w: number
  x_hr:            number[]
  base_weights:    unknown
  wrapper_weights: unknown
  expected_alpha:  number[]
}

// xs/small share the small base; medium/large share the large base.
const BASE_CTOR: Record<string, BaseNetworkCtor> = {
  xs:     EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor,
  small:  EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor,
  medium: EfficientNetLiteMattingLarge as unknown as BaseNetworkCtor,
  large:  EfficientNetLiteMattingLarge as unknown as BaseNetworkCtor,
  xl:     EfficientNetLiteMattingXL    as unknown as BaseNetworkCtor,
}

async function maxErr(backend: Backend, fx: TierFullFixture): Promise<number> {
  const x = backend.tensor(fx.canvas_h, fx.canvas_w, 4, new Float32Array(fx.x_hr))
  const model = new TierModel(
    backend, x,
    fx.base_weights as ModelWeights,
    fx.wrapper_weights as UNetWrapperWeights,
    { variant: fx.variant as UNetVariant, cHigh: fx.c_high, cLow: fx.c_low, cUp: fx.c_up },
    BASE_CTOR[fx.name],
  )
  model.run()
  const out = await backend.readback(model.output)   // 4-ch; alpha in channel 0
  const n = fx.canvas_h * fx.canvas_w
  let m = 0
  for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(out[i * 4] - fx.expected_alpha[i]))
  return m
}

const TIERS: Array<[string, TierFullFixture]> = [
  ['xs',     tierXs     as TierFullFixture],
  ['small',  tierSmall  as TierFullFixture],
  ['medium', tierMedium as TierFullFixture],
  ['large',  tierLarge  as TierFullFixture],
  ['xl',     tierXl     as TierFullFixture],
]

describe.each(BACKENDS)('TierModel base+wrapper ($name)', ({ create }) => {
  it.each(TIERS)('%s end-to-end alpha matches PyTorch', async (_name, fx) => {
    const backend = await create()
    expect(await maxErr(backend, fx)).toBeLessThan(THRESHOLD)
    backend.destroy()
  })
})
