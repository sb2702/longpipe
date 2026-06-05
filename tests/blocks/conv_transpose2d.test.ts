import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// ConvTranspose2d (gather form) — added for the optical-flow decoder (deconv +
// upflow, k4/s2/p1). Self-contained: compared against a straight-from-the-
// definition JS reference. The op consumes the canonical flat weight layout
// (mat4x4[z][o][i], M[in_sub][out_sub] = W(in,out,ky,kx)); packWeight() builds
// it from a PyTorch-shaped [in,out,kh,kw] weight exactly as pack_flow will.

const inC = 4, outC = 4, inH = 3, inW = 3, K = 4, S = 2, P = 1
const outH = (inH - 1) * S - 2 * P + K   // 6
const outW = (inW - 1) * S - 2 * P + K   // 6

const fill = (n: number, f: (i: number) => number) => Array.from({ length: n }, (_, i) => f(i))
const input = fill(inH * inW * inC, i => Math.sin(i * 1.1))            // NHWC
const Wt    = fill(inC * outC * K * K, i => Math.cos(i * 0.7) * 0.5)   // [in,out,kh,kw]
const bias  = fill(outC, i => 0.05 * (i - 1.5))

function reference(): Float32Array {
  const out = new Float32Array(outH * outW * outC)
  for (let oy = 0; oy < outH; oy++) for (let ox = 0; ox < outW; ox++) {
    for (let oc = 0; oc < outC; oc++) {
      let acc = bias[oc]
      for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
        const iyn = oy + P - ky, ixn = ox + P - kx
        if (iyn < 0 || ixn < 0 || iyn % S !== 0 || ixn % S !== 0) continue
        const iy = iyn / S, ix = ixn / S
        if (iy >= inH || ix >= inW) continue
        for (let ic = 0; ic < inC; ic++) {
          acc += input[(iy * inW + ix) * inC + ic] * Wt[((ic * outC + oc) * K + ky) * K + kx]
        }
      }
      out[(oy * outW + ox) * outC + oc] = acc
    }
  }
  return out
}

// [in,out,kh,kw] → mat4x4[z][o][i], col-major (block[col*4+row] = W(in=i*4+col, out=o*4+row)).
function packWeight(): number[] {
  const I = inC / 4, O = outC / 4
  const out = new Array(K * K * O * I * 16).fill(0)
  for (let ky = 0; ky < K; ky++) for (let kx = 0; kx < K; kx++) {
    const z = ky * K + kx
    for (let o = 0; o < O; o++) for (let i = 0; i < I; i++) {
      const block = (z * O * I + o * I + i) * 16
      for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
        const ic = i * 4 + col, oc = o * 4 + row
        out[block + col * 4 + row] = Wt[((ic * outC + oc) * K + ky) * K + kx]
      }
    }
  }
  return out
}

const THRESHOLD = 1e-3

describe.each(BACKENDS)('ConvTranspose2d ($name)', ({ create }) => {
  it('k4/s2/p1 gather matches the reference and doubles the spatial dims', async () => {
    const backend = await create()
    const inputT = backend.tensor(inH, inW, inC, new Float32Array(input))
    const op = backend.ops.ConvTranspose2d(
      inputT, { weights: packWeight(), bias },
      { outChannels: outC, kernel: K, stride: S, padding: P, activation: 'none' },
    )
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    expect(op.output.h).toBe(outH)
    expect(op.output.w).toBe(outW)
    const ref = reference()
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
