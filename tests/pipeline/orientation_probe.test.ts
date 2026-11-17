import { describe, it, expect } from 'vitest'
import { chooseOrientation, normalizedLuma, type LumaGrid } from '~/pipeline/orientation_probe'
import { orientationUvMatrix, applyUvMatrix, uprightSize, type Orientation, type Rotation } from '~/model/orientation'

// A sensor-orientation source grid with an asymmetric pattern, and the upright
// grid the <video> would show for orientation `o` (display = flip(rotCW(src))).
const SW = 32, SH = 18
function source(): LumaGrid {
  const rgba = new Uint8ClampedArray(SW * SH * 4)
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    // smooth gradient + an off-centre blob: no rotational or mirror symmetry
    const blob = Math.hypot(x - 8, y - 5) < 4 ? 120 : 0
    const v = Math.min(255, x * 4 + y * 2 + blob)
    rgba.set([v, v, v, 255], (y * SW + x) * 4)
  }
  return normalizedLuma(rgba, SW, SH)!
}
function upright(src: LumaGrid, o: Orientation): LumaGrid {
  const { w, h } = uprightSize(src.w, src.h, o)
  const m = orientationUvMatrix(o), data = new Float32Array(w * h)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const [su, sv] = applyUvMatrix(m, (x + 0.5) / w, (y + 0.5) / h)
    data[y * w + x] = src.data[Math.floor(sv * src.h) * src.w + Math.floor(su * src.w)]
  }
  return { w, h, data }
}

const ALL: Orientation[] = ([0, 90, 180, 270] as Rotation[]).flatMap(rotation => [{ rotation, flip: false }, { rotation, flip: true }])

describe('orientation probe', () => {
  it.each(ALL.map(o => [`${o.rotation}${o.flip ? '+flip' : ''}`, o] as const))('recovers %s from the upright/source pair', (_l, o) => {
    const src = source()
    const up = upright(src, o)
    const transposed = o.rotation === 90 || o.rotation === 270
    const r = chooseOrientation(up, src, transposed, { rotation: 0, flip: false })
    expect(r.orientation).toEqual(o)
  })

  it('keeps the current orientation when the evidence is not clearly better (hysteresis)', () => {
    const src = source()
    // Upright that matches 90 exactly; current = 90+flip. With noise-free data
    // 90 wins clearly, so it switches...
    expect(chooseOrientation(upright(src, { rotation: 90, flip: false }), src, true, { rotation: 90, flip: true }).orientation)
      .toEqual({ rotation: 90, flip: false })
    // ...but a mirror-symmetric scene can't decide flip: keep the current.
    const sym = new Uint8ClampedArray(SW * SH * 4)
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) { const v = x * 7; sym.set([v, v, v, 255], (y * SW + x) * 4) }
    const s = normalizedLuma(sym, SW, SH)!
    const up = upright(s, { rotation: 90, flip: false })
    expect(chooseOrientation(up, s, true, { rotation: 90, flip: true }).orientation).toEqual({ rotation: 90, flip: true })
  })

  it('flat frames are inconclusive', () => {
    const flat = new Uint8ClampedArray(16 * 4).fill(128)
    expect(normalizedLuma(flat, 4, 4)).toBeNull()
  })
})
