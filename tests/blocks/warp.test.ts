import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// Bilinear gather-warp: out[p] = sample(source, p + flowScale·flow[p].xy), edge-
// clamped. Compared against a JS bilinear reference. Constant fractional flow
// exercises the interpolation; the shift pushes some samples off-edge (clamp).

const W = 4, H = 4, C = 4
const FX = 1.3, FY = -0.7, SCALE = 1.0

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const source = Array.from({ length: H * W * C }, (_, i) => Math.sin(i * 0.5))
const flow = new Float32Array(H * W * 4)
for (let p = 0; p < H * W; p++) { flow[p * 4] = FX; flow[p * 4 + 1] = FY }

function reference(): Float32Array {
  const out = new Float32Array(H * W * C)
  const s = (xx: number, yy: number, c: number) =>
    source[(clamp(yy, 0, H - 1) * W + clamp(xx, 0, W - 1)) * C + c]
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sx = clamp(x + SCALE * FX, 0, W - 1)
    const sy = clamp(y + SCALE * FY, 0, H - 1)
    const x0 = Math.floor(sx), y0 = Math.floor(sy)
    const tx = sx - x0, ty = sy - y0
    for (let c = 0; c < C; c++) {
      const top = s(x0, y0, c) * (1 - tx) + s(x0 + 1, y0, c) * tx
      const bot = s(x0, y0 + 1, c) * (1 - tx) + s(x0 + 1, y0 + 1, c) * tx
      out[(y * W + x) * C + c] = top * (1 - ty) + bot * ty
    }
  }
  return out
}

const THRESHOLD = 1e-4

describe.each(BACKENDS)('Warp ($name)', ({ create }) => {
  it('bilinear gather matches the reference (fractional flow + edge clamp)', async () => {
    const backend = await create()
    const src  = backend.tensor(H, W, C, new Float32Array(source))
    const flowT = backend.tensor(H, W, 4, flow)
    const op = backend.ops.Warp(src, flowT, { flowScale: SCALE })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    const ref = reference()
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
