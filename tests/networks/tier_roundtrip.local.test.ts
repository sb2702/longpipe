import { describe, it, expect } from 'vitest'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { BACKENDS } from '../helpers/backends'

// Path (B): round-trip the REAL production .bin (exported by
// training/deploy/export_sdk_weights.py --ref-dir tests/fixtures/local) against
// the PyTorch final alpha — validates export → loadWeightsFromBinary → TierModel
// end-to-end on real trained weights, using the production tier_config.
//
// Artifacts are LOCAL + gitignored (real-weight .bins are 3–13 MB). The test
// self-skips when the local dir is empty (clean checkout / CI).
const THRESHOLD = 1e-3

// Vite globs resolve at transform time; ?url avoids bundling the (large) bytes.
const refUrls = (import.meta as any).glob('../fixtures/local/*.ref.json',
  { query: '?url', import: 'default', eager: true }) as Record<string, string>
const binUrls = (import.meta as any).glob('../fixtures/local/*.bin',
  { query: '?url', import: 'default', eager: true }) as Record<string, string>

const NAME_RE = /model_([a-z0-9]+)\.ref\.json$/
const tiers = Object.entries(refUrls)
  .map(([path, refUrl]) => {
    const name = path.match(NAME_RE)?.[1]
    const binUrl = Object.entries(binUrls).find(
      ([p]) => p.endsWith(`model_${name}.bin`) && !p.includes('.f16.'))?.[1]
    return { name: name!, refUrl, binUrl: binUrl! }
  })
  .filter(t => t.name && t.binUrl && TIER_CONFIG[t.name])

describe.each(BACKENDS)('production .bin round-trip ($name)', ({ create }) => {
  if (tiers.length === 0) {
    it.skip('no local fixtures — run export_sdk_weights.py --ref-dir tests/fixtures/local', () => {})
    return
  }
  it.each(tiers)('$name: .bin alpha matches PyTorch (real weights)', async ({ name, refUrl, binUrl }) => {
    const backend = await create()
    const ref = await (await fetch(refUrl)).json()
    const buf = await (await fetch(binUrl)).arrayBuffer()
    const w = loadWeightsFromBinary(buf) as any   // composite { base, wrapper, gru? }
    const cfg = TIER_CONFIG[name]

    const H = ref.canvas_h, W = ref.canvas_w
    const x = backend.tensor(H, W, 4, new Float32Array(ref.x_hr))

    let gru: { weights: any; hPrev: any } | undefined
    if (cfg.hasGru) {
      // hidden carrier sized to where the GRU runs: base res for gru-at-base.
      const hH = cfg.wrapper.gruAtBase ? ref.base_h : H
      const hW = cfg.wrapper.gruAtBase ? ref.base_w : W
      const hPrev = backend.tensor(hH, hW, 4, new Float32Array(hH * hW * 4))
      gru = { weights: w.gru, hPrev }
    }

    const model = new TierModel(backend, x, w.base, w.wrapper, cfg.wrapper, cfg.base, gru)
    model.run()

    const out = await backend.readback(model.output)
    let worst = 0
    for (let i = 0; i < H * W; i++) worst = Math.max(worst, Math.abs(out[i * 4] - ref.expected_alpha[i]))

    backend.destroy()
    expect(worst).toBeLessThan(THRESHOLD)
  })
})
