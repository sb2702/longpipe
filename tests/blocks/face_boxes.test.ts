import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// FaceBoxesFromHeatmaps — the multi-face decode. Driven by SYNTHETIC heatmaps
// (Gaussian blobs at known canonical-face configurations) rather than a trained
// fixture, because what's under test is the grouping algebra — candidates →
// eye-pair hypotheses → template matching → NMS — not the head's accuracy. Real
// heatmaps can't express "two faces at exactly these coords" on demand.
//
// The WGSL (compute, workgroup-shared candidate table) and GLSL (fragment, each
// slot recomputes everything) implementations are structurally different, so
// running the same expectations over both backends is the cross-check.

const H = 28, W = 48          // ≈ small's base/4 grid
const WIN = 3, THRESH = 0.15, BOX_SCALE = 2.4, TOL = 0.6

// Canonical template in the eye frame — must match face_boxes.wgsl tmpl_u/tmpl_w.
const T_U = [0.50002, 0.08598, 0.91410]
const T_W = [0.57150, 1.15462, 1.15462]

// Paint a face's 5 keypoints as Gaussian blobs (sigma ~0.8 cells, the training
// render's scale) into an 8-ch NHWC heatmap buffer.
function paintFace(hm: Float32Array, eyeL: [number, number], interocular: number, roll: number, peak = 0.9) {
  const ex: [number, number] = [Math.cos(roll), Math.sin(roll)]
  const ey: [number, number] = [-ex[1], ex[0]]
  const pts: Array<[number, number]> = [
    eyeL,
    [eyeL[0] + interocular * ex[0], eyeL[1] + interocular * ex[1]],
  ]
  for (let t = 0; t < 3; t++) {
    pts.push([
      eyeL[0] + interocular * (T_U[t] * ex[0] + T_W[t] * ey[0]),
      eyeL[1] + interocular * (T_U[t] * ex[1] + T_W[t] * ey[1]),
    ])
  }
  pts.forEach(([px, py], k) => {
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const d2 = (c - px) ** 2 + (r - py) ** 2
      const v = peak * Math.exp(-d2 / (2 * 0.8 * 0.8))
      const i = (r * W + c) * 8 + k
      hm[i] = Math.max(hm[i], v)
    }
  })
  return pts
}

const readBoxes = (b: Float32Array, k: number) =>
  Array.from({ length: k }, (_, i) => ({
    cx: b[i * 4], cy: b[i * 4 + 1], hs: b[i * 4 + 2], score: b[i * 4 + 3],
  }))

async function decode(create: () => Promise<any>, hm8: Float32Array, maxFaces: number) {
  const backend = await create()
  const hm = backend.tensor(H, W, 8, hm8)
  const op = backend.ops.FaceBoxesFromHeatmaps(hm, {
    win: WIN, thresh: THRESH, boxScale: BOX_SCALE, maxFaces, tol: TOL,
  })
  op.run()
  const out = await backend.readback(op.output)
  backend.destroy()
  return readBoxes(out, maxFaces)
}

describe.each(BACKENDS)('FaceBoxesFromHeatmaps ($name)', ({ create }) => {
  it('finds two separated faces and puts each box on its own face', async () => {
    const hm = new Float32Array(H * W * 8)
    paintFace(hm, [8, 8], 5, 0, 0.9)      // left face,  interocular 5 cells
    paintFace(hm, [30, 14], 4, 0, 0.8)    // right face, interocular 4 cells
    const boxes = await decode(create, hm, 4)

    const live = boxes.filter(b => b.score > 0)
    expect(live.length).toBe(2)

    // Face centers: hull midpoint of the 5 painted keypoints, in frame fractions.
    // Sorted by cx so the assertion doesn't depend on NMS ordering.
    const cxs = live.map(b => b.cx).sort((a, b) => a - b)
    expect(cxs[0]).toBeCloseTo((8 + 5 * 0.5 + 0.5) / W, 1)    // left face ≈ x 10.5
    expect(cxs[1]).toBeCloseTo((30 + 4 * 0.5 + 0.5) / W, 1)   // right face ≈ x 32
    // Boxes must be disjoint — the single-face decode's failure is one hull
    // spanning both, which would put a single cx between the two.
    expect(cxs[1] - cxs[0]).toBeGreaterThan(0.3)
  })

  it('scales each box to its own face (the bigger face gets the bigger box)', async () => {
    const hm = new Float32Array(H * W * 8)
    paintFace(hm, [6, 9], 7, 0, 0.9)      // near face
    paintFace(hm, [34, 12], 3, 0, 0.85)   // far face
    const boxes = (await decode(create, hm, 4)).filter(b => b.score > 0)
    expect(boxes.length).toBe(2)
    const [near, far] = boxes.sort((a, b) => a.cx - b.cx)
    expect(near.hs).toBeGreaterThan(far.hs * 1.5)
  })

  it('one face fills exactly one slot; the rest are zeroed', async () => {
    const hm = new Float32Array(H * W * 8)
    paintFace(hm, [20, 10], 5, 0)
    const boxes = await decode(create, hm, 4)
    expect(boxes[0].score).toBeGreaterThan(THRESH)
    for (let i = 1; i < 4; i++) {
      expect(boxes[i]).toEqual({ cx: 0, cy: 0, hs: 0, score: 0 })
    }
  })

  it('empty heatmaps → no faces', async () => {
    const boxes = await decode(create, new Float32Array(H * W * 8), 4)
    expect(boxes.every(b => b.score === 0)).toBe(true)
  })

  it('rejects a cross-paired hypothesis (two faces, no false third)', async () => {
    // Two faces side by side: face A's R-eye and face B's L-eye are adjacent, so
    // the (A.L-eye, B.R-eye) and (B.L-eye, A.R-eye) pairs are both geometrically
    // available. They must lose — the implied interocular is wrong and there's
    // no nose/mouth support at the predicted spots.
    const hm = new Float32Array(H * W * 8)
    paintFace(hm, [10, 10], 4, 0)
    paintFace(hm, [22, 10], 4, 0)
    const boxes = (await decode(create, hm, 4)).filter(b => b.score > 0)
    expect(boxes.length).toBe(2)
    const cxs = boxes.map(b => b.cx).sort((a, b) => a - b)
    expect(cxs[0]).toBeCloseTo((10 + 2 + 0.5) / W, 1)
    expect(cxs[1]).toBeCloseTo((22 + 2 + 0.5) / W, 1)
  })

  it('honors maxFaces (3 faces present, K=2 keeps the two best)', async () => {
    const hm = new Float32Array(H * W * 8)
    paintFace(hm, [4, 8],  4, 0, 0.9)
    paintFace(hm, [20, 8], 4, 0, 0.8)
    paintFace(hm, [36, 8], 4, 0, 0.4)   // weakest — should be the one dropped
    const boxes = await decode(create, hm, 2)
    expect(boxes.length).toBe(2)
    expect(boxes.every(b => b.score > 0)).toBe(true)
    // The 0.4-peak face must not be among the survivors.
    expect(boxes.every(b => b.cx < (30 + 0.5) / W)).toBe(true)
  })

  it('tracks a rolled face (the eye frame carries rotation)', async () => {
    const hm = new Float32Array(H * W * 8)
    const roll = 0.35   // ~20°
    const pts = paintFace(hm, [18, 9], 5, roll)
    const boxes = (await decode(create, hm, 4)).filter(b => b.score > 0)
    expect(boxes.length).toBe(1)
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1])
    const cx = ((Math.min(...xs) + Math.max(...xs)) / 2 + 0.5) / W
    const cy = ((Math.min(...ys) + Math.max(...ys)) / 2 + 0.5) / H
    expect(boxes[0].cx).toBeCloseTo(cx, 1)
    expect(boxes[0].cy).toBeCloseTo(cy, 1)
  })
})
