import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import type { FlowWeights } from '~/model/weights'

// Structural smoke for the flow head across encoder shapes: synthetic weights at
// real tier dims. Includes XL (lite3) — no trained lite3 flow checkpoint yet, so
// it's smoke-only here; fidelity (real weights) is covered for lite0 tiers in
// optical_flow_fidelity.test.ts.

const DEC_W = 16
const KS = [5, 5, 3, 3]
const TIERS = [
  { name: 'medium-lite0', taps: [24, 40, 112, 320], baseH: 160, baseW: 256 },
  { name: 'xl-lite3',     taps: [32, 48, 136, 384], baseH: 192, baseW: 320 },
]

const convOut = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1
const convW = (inC: number, outC: number, k: number) => ({
  weights: Float32Array.from({ length: k * k * (outC / 4) * (inC / 4) * 16 }, (_, i) => Math.sin(i * 0.017) * 0.05),
  bias:    new Float32Array(outC),
})
const sinFill = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01))

function buildWeights(taps: number[]): FlowWeights {
  const fusedC   = taps.map(t => DEC_W + t)
  const stageInC = taps.map((_, i) => (i === 0 ? DEC_W : fusedC[i - 1]))
  const catC     = fusedC.map(f => f + DEC_W + 4)
  return {
    stem:       convW(8, DEC_W, 7),
    stages:     taps.map((_, i) => convW(stageInC[i], DEC_W, KS[i])),
    predictBot: convW(fusedC[fusedC.length - 1], 4, 3),
    // resize-conv deconv: k3 conv applied after a parameter-free bilinear 2× upsample
    deconv:     taps.slice(1).map((_, j) => convW(j === 0 ? fusedC[fusedC.length - 1] : catC[catC.length - 1 - j], DEC_W, 3)),
    predict:    taps.slice(1).map((_, j) => convW(catC[catC.length - 2 - j], 4, 3)),
  }
}

function stageSizes(taps: number[], baseH: number, baseW: number): Array<[number, number]> {
  const out: Array<[number, number]> = []
  let h = convOut(baseH, 7, 2, 3), w = convOut(baseW, 7, 2, 3)
  for (let i = 0; i < taps.length; i++) {
    const p = (KS[i] - 1) / 2
    h = convOut(h, KS[i], 2, p); w = convOut(w, KS[i], 2, p)
    out.push([h, w])
  }
  return out
}

describe.each(BACKENDS)('OpticalFlowNet ($name)', ({ create }) => {
  for (const tier of TIERS) {
    it(`${tier.name}: builds + runs, emits base/4 4-ch flow`, async () => {
      const backend = await create()
      const weights = buildWeights(tier.taps)
      const hw = stageSizes(tier.taps, tier.baseH, tier.baseW)

      const frameA = backend.tensor(tier.baseH, tier.baseW, 4, sinFill(tier.baseH * tier.baseW * 4))
      const frameB = backend.tensor(tier.baseH, tier.baseW, 4, sinFill(tier.baseH * tier.baseW * 4))
      const taps = tier.taps.map((c, i) => backend.tensor(hw[i][0], hw[i][1], c, sinFill(hw[i][0] * hw[i][1] * c)))

      const net = new OpticalFlowNet(backend, frameA, frameB, taps, weights, DEC_W)
      net.run()
      const out = await backend.readback(net.output)
      backend.destroy()

      expect(net.output.h).toBe(hw[0][0])   // base/4
      expect(net.output.w).toBe(hw[0][1])
      expect(net.output.c).toBe(4)
      expect(out.every(Number.isFinite)).toBe(true)
    })
  }
})
