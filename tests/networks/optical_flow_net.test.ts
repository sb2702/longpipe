import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import type { FlowWeights } from '~/model/weights'

// Structural smoke test for the flow head: synthetic weights, real medium dims
// (base 144×256 — its /32 level rounds 9→5 so the decoder's 2× deconv overshoots
// the skip by a pixel, exercising matchSize). Fidelity vs PyTorch is step 4.

const decW = 16
const TAPC = [24, 40, 112, 320]              // full-encoder taps (medium/large)
const KS = [5, 5, 3, 3]
const baseH = 144, baseW = 256

const convOut = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1
const convW = (inC: number, outC: number, k: number) => ({
  weights: Float32Array.from({ length: k * k * (outC / 4) * (inC / 4) * 16 }, (_, i) => Math.sin(i * 0.017) * 0.05),
  bias:    new Float32Array(outC),
})

const fusedC   = TAPC.map(t => decW + t)                            // [40,56,128,336]
const stageInC = TAPC.map((_, i) => (i === 0 ? decW : fusedC[i - 1]))  // [16,40,56,128]
const catC     = fusedC.map(f => f + decW + 4)                     // [60,76,148,360]

const weights: FlowWeights = {
  stem:       convW(8, decW, 7),
  stages:     TAPC.map((_, i) => convW(stageInC[i], decW, KS[i])),
  predictBot: convW(fusedC[3], 4, 3),
  deconv:     [convW(fusedC[3], decW, 4), convW(catC[2], decW, 4), convW(catC[1], decW, 4)],
  upflow:     [convW(4, 4, 4), convW(4, 4, 4), convW(4, 4, 4)],
  predict:    [convW(catC[2], 4, 3), convW(catC[1], 4, 3), convW(catC[0], 4, 3)],
}

// stage output sizes — taps are placed here so the fusion concat lines up.
const stageHW: Array<[number, number]> = []
{
  let h = convOut(baseH, 7, 2, 3), w = convOut(baseW, 7, 2, 3)
  for (let i = 0; i < TAPC.length; i++) {
    const p = (KS[i] - 1) / 2
    h = convOut(h, KS[i], 2, p); w = convOut(w, KS[i], 2, p)
    stageHW.push([h, w])
  }
}

const sinFill = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.01))

describe.each(BACKENDS)('OpticalFlowNet ($name)', ({ create }) => {
  it('builds + runs, emits base/4 4-ch flow with finite values', async () => {
    const backend = await create()
    const frameA = backend.tensor(baseH, baseW, 4, sinFill(baseH * baseW * 4))
    const frameB = backend.tensor(baseH, baseW, 4, sinFill(baseH * baseW * 4))
    const taps = TAPC.map((c, i) =>
      backend.tensor(stageHW[i][0], stageHW[i][1], c, sinFill(stageHW[i][0] * stageHW[i][1] * c)))

    const net = new OpticalFlowNet(backend, frameA, frameB, taps, weights, decW)
    net.run()
    const out = await backend.readback(net.output)
    backend.destroy()

    expect(net.output.h).toBe(stageHW[0][0])   // base/4 height
    expect(net.output.w).toBe(stageHW[0][1])   // base/4 width
    expect(net.output.c).toBe(4)               // flow in .xy
    expect(out.every(Number.isFinite)).toBe(true)
  })
})
