import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// Flow-gated stabilizer. Verifies the full chain vs a JS reference: peak-hold
// envelope (release decay), soft gate with leak floor, and the pred/ref blend.
// Output packs the blended alpha in .x and the new envelope in .y.

const W = 4, H = 4, N = W * H
const T_LO = 1.0, T_HI = 3.0, LEAK = 0.2, RELEASE = 0.9
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

// Per-pixel inputs (deterministic, varied so each branch is exercised).
const fx      = (p: number) => (p % 5) * 0.8       // flow magnitude 0 .. 3.2
const predX   = (p: number) => Math.sin(p * 0.4)
const refX    = (p: number) => Math.cos(p * 0.3)
const envPrevY = (p: number) => (p % 3) * 1.5      // 0, 1.5, 3.0

function pack(setter: (p: number, q: Float32Array) => void): Float32Array {
  const a = new Float32Array(N * 4)
  for (let p = 0; p < N; p++) setter(p, a.subarray(p * 4, p * 4 + 4))
  return a
}

describe.each(BACKENDS)('Stabilize ($name)', ({ create }) => {
  it('envelope + gate + blend match the reference', async () => {
    const backend = await create()
    const flow = backend.tensor(H, W, 4, pack((p, q) => { q[0] = fx(p) }))
    const pred = backend.tensor(H, W, 4, pack((p, q) => { q[0] = predX(p) }))
    const ref  = backend.tensor(H, W, 4, pack((p, q) => { q[0] = refX(p) }))
    const env  = backend.tensor(H, W, 4, pack((p, q) => { q[1] = envPrevY(p) }))

    const op = backend.ops.Stabilize(flow, pred, ref, env,
      { tLo: T_LO, tHi: T_HI, leak: LEAK, release: RELEASE })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    let maxErr = 0
    for (let p = 0; p < N; p++) {
      const mag = Math.abs(fx(p))
      const envE = Math.max(mag, RELEASE * envPrevY(p))
      let g = clamp((envE - T_LO) / Math.max(T_HI - T_LO, 1e-3), 0, 1)
      g = Math.max(g, LEAK)
      const stabE = g * predX(p) + (1 - g) * refX(p)
      maxErr = Math.max(maxErr, Math.abs(got[p * 4] - stabE), Math.abs(got[p * 4 + 1] - envE))
    }
    expect(maxErr).toBeLessThan(1e-4)
  })
})
