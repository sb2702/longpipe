import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// backend.copyTensor: GPU-resident tensor→tensor copy used to thread recurrent
// ConvGRU state across frames. Validates an exact byte-for-byte copy on both
// backends, plus that src and dst are independent buffers afterward (mutating
// src does not retroactively change dst).
describe.each(BACKENDS)('copyTensor ($name)', ({ create }) => {
  it('copies tensor contents exactly', async () => {
    const backend = await create()
    const H = 6, W = 5, C = 4
    const data = new Float32Array(H * W * C)
    for (let i = 0; i < data.length; i++) data[i] = (i * 7 + 3) % 251 - 125  // mixed sign, non-trivial

    const src = backend.tensor(H, W, C, data)
    const dst = backend.tensor(H, W, C)   // uninitialized

    backend.copyTensor(src, dst)

    const out = await backend.readback(dst)
    let worst = 0
    for (let i = 0; i < data.length; i++) worst = Math.max(worst, Math.abs(out[i] - data[i]))
    backend.destroy()
    expect(worst).toBe(0)
  })

  it('produces an independent buffer (later src writes do not leak into dst)', async () => {
    const backend = await create()
    const H = 4, W = 4, C = 4
    const first  = new Float32Array(H * W * C).fill(1)
    const second = new Float32Array(H * W * C).fill(9)

    const src = backend.tensor(H, W, C, first)
    const dst = backend.tensor(H, W, C)
    backend.copyTensor(src, dst)

    // Overwrite src with a fresh upload; dst must still hold the first values.
    const src2 = backend.tensor(H, W, C, second)
    backend.copyTensor(src2, src)   // src now holds `second`

    const out = await backend.readback(dst)
    backend.destroy()
    expect(out.every(v => v === 1)).toBe(true)
  })

  it('throws on size mismatch', async () => {
    const backend = await create()
    const a = backend.tensor(4, 4, 4)
    const b = backend.tensor(2, 4, 4)
    expect(() => backend.copyTensor(a, b)).toThrow()
    backend.destroy()
  })
})
