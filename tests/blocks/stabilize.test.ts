import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// Flow-gated stabilizer. Verifies the full chain vs a JS reference: peak-hold
// envelope (release decay), soft gate with leak floor, and the pred/ref blend.
// Output packs the blended alpha in .x and the new envelope in .y.

const W = 4, H = 4, N = W * H
const T_LO = 1.0, T_HI = 3.0, LEAK = 0.2, RELEASE = 0.9
const T_DIV = 0.5, DIV_SCALE = 1.0, STEP_X = 1, STEP_Y = 1
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

// Per-pixel inputs (deterministic, varied so each branch is exercised).
const fx      = (p: number) => (p % 5) * 0.8       // flow x 0 .. 3.2
const fy      = (p: number) => ((p % 4) - 1.5) * 0.7  // flow y, signed
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
    const flow = backend.tensor(H, W, 4, pack((p, q) => { q[0] = fx(p); q[1] = fy(p) }))
    const pred = backend.tensor(H, W, 4, pack((p, q) => { q[0] = predX(p) }))
    const ref  = backend.tensor(H, W, 4, pack((p, q) => { q[0] = refX(p) }))
    const env  = backend.tensor(H, W, 4, pack((p, q) => { q[1] = envPrevY(p) }))

    const op = backend.ops.Stabilize(flow, pred, ref, env,
      { tLo: T_LO, tHi: T_HI, leak: LEAK, release: RELEASE,
        tDiv: T_DIV, divScale: DIV_SCALE, stepX: STEP_X, stepY: STEP_Y })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    let maxErr = 0
    for (let p = 0; p < N; p++) {
      const x = p % W, y = Math.floor(p / W)
      const envE = Math.max(Math.hypot(fx(p), fy(p)), RELEASE * envPrevY(p))
      // divergence: ±step finite-difference, edge-clamped
      const xr = Math.min(x + STEP_X, W - 1), xl = Math.max(x - STEP_X, 0)
      const yd = Math.min(y + STEP_Y, H - 1), yu = Math.max(y - STEP_Y, 0)
      const dfx = fx(y * W + xr) - fx(y * W + xl)
      const dfy = fy(yd * W + x) - fy(yu * W + x)
      const divg = Math.abs(dfx + dfy)
      const gMag = clamp((envE - T_LO) / Math.max(T_HI - T_LO, 1e-3), 0, 1)
      const gDiv = clamp((divg - T_DIV) / Math.max(DIV_SCALE, 1e-3), 0, 1)
      const g = Math.max(Math.max(gMag, gDiv), LEAK)
      const stabE = g * predX(p) + (1 - g) * refX(p)
      maxErr = Math.max(maxErr, Math.abs(got[p * 4] - stabE), Math.abs(got[p * 4 + 1] - envE))
    }
    expect(maxErr).toBeLessThan(1e-4)
  })
})
