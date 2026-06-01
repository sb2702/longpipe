import { describe, it, expect } from 'vitest'
import type { ModelWeights, UNetWrapperWeights } from '~/model/weights'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { TierModel, type BaseNetworkCtor } from '~/model/tier_model'
import type { UNetVariant } from '~/model/blocks/unet_wrapper'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small'
import { BACKENDS } from '../helpers/backends'

import fx from '../fixtures/tier_full_small.json'
import binUrl from '../fixtures/tier_small.bin?url'

// Weight-serialization round-trip: the small tier's composite weights (base +
// wrapper, with the native packings — conv_expand mat4x2, up_final split, etc.)
// are packed to .bin by serialize_weights_binary, loaded by loadWeightsFromBinary
// (structure-agnostic: number[] leaves → Float32Array views), and run through
// TierModel. Result must match the JSON-weights alpha (same deterministic model).
const THRESHOLD = 1e-3

describe.each(BACKENDS)('Weight .bin round-trip ($name)', ({ create }) => {
  it('TierModel from loaded .bin matches the JSON-weights alpha', async () => {
    const buf = await (await fetch(binUrl)).arrayBuffer()
    const w = loadWeightsFromBinary(buf) as unknown as {
      base: ModelWeights; wrapper: UNetWrapperWeights
    }

    const backend = await create()
    const x = backend.tensor(fx.canvas_h, fx.canvas_w, 4, new Float32Array(fx.x_hr))
    const model = new TierModel(
      backend, x, w.base, w.wrapper,
      { variant: fx.variant as UNetVariant, cHigh: fx.c_high, cLow: fx.c_low, cUp: fx.c_up },
      EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor,
    )
    model.run()
    const out = await backend.readback(model.output)
    backend.destroy()

    let worst = 0
    const n = fx.canvas_h * fx.canvas_w
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(out[i * 4] - fx.expected_alpha[i]))
    expect(worst).toBeLessThan(THRESHOLD)
  })
})
