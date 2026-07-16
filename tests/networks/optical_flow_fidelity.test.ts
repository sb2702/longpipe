import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net'
import flow_medium from '../fixtures/flow_medium.json'
import flow_small from '../fixtures/flow_small.json'
import flow_large from '../fixtures/flow_large.json'
import flow_xs from '../fixtures/flow_xs.json'
import flow_xl from '../fixtures/flow_xl.json'

// Fidelity: the SDK OpticalFlowNet vs the real trained WrapperFlowNet (relu6 +
// resize-conv decoder + backward flow) on the same packed weights + taps + frame
// pair. Exact agreement validates pack_flow + the whole op graph (Conv2d-relu6,
// BilinearUpsample, ChannelConcat, Crop) end-to-end.
const FIXTURES = [
  { name: 'medium', fx: flow_medium as any },   // full encoder (lite0), variant A
  { name: 'small',  fx: flow_small  as any },   // small encoder (3 taps), variant A
  { name: 'large',  fx: flow_large  as any },   // full encoder (lite0), variant D
  { name: 'xs',     fx: flow_xs     as any },   // small encoder, tap-half (base/2)
  { name: 'xl',     fx: flow_xl     as any },   // full encoder (lite3), variant E
]

describe.each(BACKENDS)('OpticalFlowNet fidelity ($name)', ({ name: backendName, create }) => {
  for (const { name, fx } of FIXTURES) {
    it(`${name}: base/4 flow matches the PyTorch reference`, async () => {
      const backend = await create()
      const frameA = backend.tensor(fx.baseH, fx.baseW, 4, new Float32Array(fx.frameA))
      const frameB = backend.tensor(fx.baseH, fx.baseW, 4, new Float32Array(fx.frameB))
      const taps = fx.taps.map((t: number[], i: number) => {
        const [h, w, c] = fx.tapShapes[i]
        return backend.tensor(h, w, c, new Float32Array(t))
      })

      const opts: { fuseStem?: boolean; halfTap?: any } = {}
      if (fx.fuseStem) {
        const [hh, hw, hc] = fx.halfTapShape
        opts.fuseStem = true
        opts.halfTap = backend.tensor(hh, hw, hc, new Float32Array(fx.halfTap))
      }
      const net = new OpticalFlowNet(backend, frameA, frameB, taps, fx.flowWeights, fx.decW, opts)
      net.run()
      const got = await backend.readback(net.output)
      backend.destroy()

      const [eh, ew] = fx.expectedHW
      expect(net.output.h).toBe(eh)
      expect(net.output.w).toBe(ew)

      let maxErr = 0, sumSq = 0, sigSq = 0
      const n = eh * ew
      for (let p = 0; p < n; p++) {
        for (let c = 0; c < 2; c++) {
          const d = got[p * 4 + c] - fx.expected[p * 2 + c]
          maxErr = Math.max(maxErr, Math.abs(d))
          sumSq += d * d
          sigSq += fx.expected[p * 2 + c] ** 2
        }
      }
      const rmse = Math.sqrt(sumSq / (n * 2))
      const sigRms = Math.sqrt(sigSq / (n * 2))
      console.log(`[${backendName}] ${name}: maxErr=${maxErr.toFixed(4)} rmse=${rmse.toFixed(4)} signalRms=${sigRms.toFixed(4)}`)

      expect(maxErr).toBeLessThan(0.02)   // f32 accumulation across the deep net; reference is exact
    })
  }
})
