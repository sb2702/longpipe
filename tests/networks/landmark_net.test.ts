import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { LandmarkNet } from '~/model/networks/landmark_net'
import landmark_mesh from '../fixtures/landmark_mesh.json'

// Fidelity: the SDK LandmarkNet vs the real trained checkpoint
// (checkpoints_mesh_w5_c_32, val_NME 0.0239) on the same packed weights +
// seeded input (training/deploy/gen_landmark_fixture.py). Exact agreement
// validates the BN-fused packing, the dense conv chain, and the fc-as-valid-
// conv trick end-to-end.
//
// Output coords are normalized to [0,1] of the crop. Threshold 5e-3 = ~1.3 px
// at the 256 crop: WebGPU lands ~1e-6; WebGL-via-ANGLE accumulates ~1e-3 of
// fp32 drift over the 13-conv chain (same profile as the tier fidelity tests),
// both far below the model's own val_NME (0.024 IOD ≈ several px).
const fx = landmark_mesh as any

describe.each(BACKENDS)('LandmarkNet ($name)', ({ name: backendName, create }) => {
  it('predicts the PyTorch reference landmarks on the trained checkpoint', async () => {
    const backend = await create()
    const input = backend.tensor(fx.crop, fx.crop, 4, new Float32Array(fx.input))
    const net = new LandmarkNet(backend, input, fx.weights)
    net.run()
    const got = await backend.readback(net.output)
    backend.destroy()

    expect(net.output.h).toBe(1)
    expect(net.output.w).toBe(1)
    expect(net.output.c).toBe(fx.numPts * 2)

    let maxErr = 0, sumSq = 0
    for (let i = 0; i < fx.expected.length; i++) {
      const d = got[i] - fx.expected[i]
      maxErr = Math.max(maxErr, Math.abs(d))
      sumSq += d * d
    }
    const rmse = Math.sqrt(sumSq / fx.expected.length)
    console.log(`[${backendName}] landmark_mesh: maxErr=${maxErr.toExponential(2)} rmse=${rmse.toExponential(2)}`)

    expect(maxErr).toBeLessThan(5e-3)
  })

  it('emits finite coords roughly inside the crop', async () => {
    const backend = await create()
    const input = backend.tensor(fx.crop, fx.crop, 4, new Float32Array(fx.input))
    const net = new LandmarkNet(backend, input, fx.weights)
    net.run()
    const got = await backend.readback(net.output)
    backend.destroy()

    expect(Array.from(got).every(Number.isFinite)).toBe(true)
    // Trained model on in-range input: predictions live in (loosely) [0,1].
    for (let i = 0; i < fx.numPts * 2; i++) {
      expect(got[i]).toBeGreaterThan(-0.5)
      expect(got[i]).toBeLessThan(1.5)
    }
  })
})
