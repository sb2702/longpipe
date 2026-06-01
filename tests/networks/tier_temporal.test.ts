import { describe, it, expect } from 'vitest'
import type { Backend, Tensor } from '~/model/backend'
import type { ModelWeights, UNetWrapperWeights, ConvGRUWeights } from '~/model/weights'
import { TierModel, type BaseNetworkCtor } from '~/model/tier_model'
import type { UNetVariant } from '~/model/blocks/unet_wrapper'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small'
import { BACKENDS } from '../helpers/backends'

import fx from '../fixtures/tier_temporal_small.json'

// Multi-frame: the output GRU threads hidden state across frames via the option-A
// carrier — frame t's GRU output (a, b_out, h_new, 0) is fed back as frame t+1's
// hPrev (hidden read from .z). Reference = PyTorch per-frame loop.
const THRESHOLD = 1e-3

describe.each(BACKENDS)('TierModel temporal GRU ($name)', ({ create }) => {
  it('multi-frame alpha matches PyTorch with hPrev threaded', async () => {
    const backend = await create()
    const H = fx.canvas_h, W = fx.canvas_w
    const params = { variant: fx.variant as UNetVariant, cHigh: fx.c_high, cLow: fx.c_low, cUp: fx.c_up }
    const gruW = fx.gru_weights as unknown as ConvGRUWeights

    // Frame 0 hidden = zero carrier.
    let hPrev: Tensor = backend.tensor(H, W, 4, new Float32Array(H * W * 4))
    let worst = 0

    for (let t = 0; t < fx.frames; t++) {
      const x = backend.tensor(H, W, 4, new Float32Array(fx.x_hr_per_frame[t]))
      const model = new TierModel(
        backend, x,
        fx.base_weights as unknown as ModelWeights,
        fx.wrapper_weights as unknown as UNetWrapperWeights,
        params,
        EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor,
        { weights: gruW, hPrev },
      )
      model.run()

      const out = await backend.readback(model.output)   // alpha in channel 0
      const ref = fx.expected_alphas[t]
      for (let i = 0; i < H * W; i++) worst = Math.max(worst, Math.abs(out[i * 4] - ref[i]))

      hPrev = model.hiddenState!   // GRU carrier → next frame's hPrev
    }

    backend.destroy()
    expect(worst).toBeLessThan(THRESHOLD)
  })
})
