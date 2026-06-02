import { describe, it, expect } from 'vitest'
import { autotunePreset } from '~/pipeline/worker/autotune'
import { TIER_CONFIG } from '~/model/tier_config'
import { BACKENDS } from '../helpers/backends'

// Regression: autotune builds each tier's BASE network with synthesized
// (zero-filled, shape-correct) weights via synthBackend. If synthBackend fails
// to cover a weight-bearing op the base nets use (it previously missed
// ProjResidual + ConcatConv2d), the first microbench throws and autotune
// returns no preset → "no implementable preset available". This exercises the
// full synth-bench path on a real backend and asserts a valid tier comes back.
describe.each(BACKENDS)('autotunePreset ($name)', ({ create }) => {
  it('benches all tiers without throwing and picks an implementable preset', async () => {
    const backend = await create()
    const preset = await autotunePreset(backend, 30)
    backend.destroy()
    expect(preset).toBeTruthy()
    expect(TIER_CONFIG[preset.model], `picked ${preset.model}`).toBeDefined()
  })
})
