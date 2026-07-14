import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import face_small from '../fixtures/face_small.json'

// FaceBoxFromHeatmaps + CropResample — the GPU-resident landmark front-end.
// The box op is diffed against a JS reference of the decode contract
// (windowed soft-argmax centroid, hull, pixel-square box) on the REAL trained
// heatmaps from the face_small fixture; the crop op against a JS bilinear
// reference on a synthetic gradient frame.

const fx = face_small as any
const WIN = 3, THRESH = 0.15, BOX_SCALE = 2.4

function refBox(hm5: number[], h: number, w: number) {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, score = 1e9
  for (let k = 0; k < 5; k++) {
    let peak = -1, pr = 0, pc = 0
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const v = hm5[(r * w + c) * 5 + k]
      if (v > peak) { peak = v; pr = r; pc = c }
    }
    const r0 = Math.max(0, pr - WIN), r1 = Math.min(h, pr + WIN + 1)
    const c0 = Math.max(0, pc - WIN), c1 = Math.min(w, pc + WIN + 1)
    let wsum = 0, sy = 0, sx = 0
    for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
      const v = hm5[(r * w + c) * 5 + k]
      wsum += v; sy += v * r; sx += v * c
    }
    wsum = Math.max(wsum, 1e-6)
    const kx = sx / wsum, ky = sy / wsum
    x0 = Math.min(x0, kx); x1 = Math.max(x1, kx)
    y0 = Math.min(y0, ky); y1 = Math.max(y1, ky)
    score = Math.min(score, peak)
  }
  if (score < THRESH) score = 0
  const halfPx = 0.5 * BOX_SCALE * Math.max(x1 - x0, y1 - y0)
  return {
    cx: ((x0 + x1) / 2 + 0.5) / w,
    cy: ((y0 + y1) / 2 + 0.5) / h,
    hs: halfPx / w,
    score,
  }
}

describe.each(BACKENDS)('FaceBoxFromHeatmaps ($name)', ({ create }) => {
  it('matches the JS soft-argmax reference on real heatmaps', async () => {
    const backend = await create()
    const [h, w] = fx.expectedHW
    // Fixture heatmaps are 5-ch NHWC; pad to the 8-ch tensor FaceHeatmapNet emits.
    const hm8 = new Float32Array(h * w * 8)
    for (let p = 0; p < h * w; p++)
      for (let k = 0; k < 5; k++) hm8[p * 8 + k] = fx.expected[p * 5 + k]
    const hm = backend.tensor(h, w, 8, hm8)

    const op = backend.ops.FaceBoxFromHeatmaps(hm, { win: WIN, thresh: THRESH, boxScale: BOX_SCALE })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    const ref = refBox(fx.expected, h, w)
    expect(ref.score).toBeGreaterThan(0.3)               // fixture has a real face
    expect(Math.abs(got[0] - ref.cx)).toBeLessThan(1e-4)
    expect(Math.abs(got[1] - ref.cy)).toBeLessThan(1e-4)
    expect(Math.abs(got[2] - ref.hs)).toBeLessThan(1e-4)
    expect(Math.abs(got[3] - ref.score)).toBeLessThan(1e-4)
  })

  it('zeroes the score below threshold', async () => {
    const backend = await create()
    const [h, w] = fx.expectedHW
    const hm = backend.tensor(h, w, 8, new Float32Array(h * w * 8).fill(0.01))
    const op = backend.ops.FaceBoxFromHeatmaps(hm, { win: WIN, thresh: THRESH, boxScale: BOX_SCALE })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()
    expect(got[3]).toBe(0)
  })
})

describe.each(BACKENDS)('CropResample ($name)', ({ create }) => {
  it('matches a JS bilinear reference (crop + ImageNet normalize)', async () => {
    const backend = await create()
    const H = 56, W = 96, OUT = 32
    const MEAN: [number, number, number] = [0.485, 0.456, 0.406]
    const STD:  [number, number, number] = [0.229, 0.224, 0.225]
    // Smooth gradient frame so bilinear samples are non-trivial but exact.
    const frame = new Float32Array(H * W * 4)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4
      frame[i] = x / (W - 1); frame[i + 1] = y / (H - 1); frame[i + 2] = (x + y) / (W + H - 2)
    }
    const box = { cx: 0.55, cy: 0.48, hs: 0.18, score: 0.9 }   // fractions, hs of width
    const ft = backend.tensor(H, W, 4, frame)
    const bt = backend.tensor(1, 1, 4, new Float32Array([box.cx, box.cy, box.hs, box.score]))
    const op = backend.ops.CropResample(ft, bt, { outH: OUT, outW: OUT, mean: MEAN, std: STD })
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()

    const samp = (x: number, y: number, c: number) =>
      frame[(Math.min(Math.max(y, 0), H - 1) * W + Math.min(Math.max(x, 0), W - 1)) * 4 + c]
    let maxErr = 0
    const cx = box.cx * W, cy = box.cy * H, side = 2 * box.hs * W
    for (let y = 0; y < OUT; y++) for (let x = 0; x < OUT; x++) {
      const sx = Math.min(Math.max(cx + (x - OUT / 2) * side / OUT, 0), W - 1)
      const sy = Math.min(Math.max(cy + (y - OUT / 2) * side / OUT, 0), H - 1)
      const x0 = Math.floor(sx), y0 = Math.floor(sy)
      const tx = sx - x0, ty = sy - y0
      for (let c = 0; c < 3; c++) {
        const top = samp(x0, y0, c) * (1 - tx) + samp(x0 + 1, y0, c) * tx
        const bot = samp(x0, y0 + 1, c) * (1 - tx) + samp(x0 + 1, y0 + 1, c) * tx
        const ref = (top * (1 - ty) + bot * ty - MEAN[c]) / STD[c]
        maxErr = Math.max(maxErr, Math.abs(got[(y * OUT + x) * 4 + c] - ref))
      }
    }
    expect(maxErr).toBeLessThan(1e-4)
  })
})
