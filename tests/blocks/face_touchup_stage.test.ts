import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import type { FaceTopology } from '~/model/backend'
import topoJson from '../fixtures/face_topology.json'

// FaceTouchupStage — the Tensor→Tensor form of the touch-up (the composable
// shape for the one-compositor architecture). Invariant tests:
//   1. no face (box score 0)   → output === input (exact passthrough)
//   2. strength 0, face present → output ≈ input (mesh redraws original pixels)
//   3. strength 1, heavy blur  → pixels change inside the face box, and are
//      untouched far outside it (all-white weight mask, so the whole mesh
//      region is eligible — the box hull bounds the change).
// Landmarks are set to the canonical mesh UVs, so the mesh is an affine map of
// the canonical face — non-degenerate triangles without a real model run.

const H = 96, W = 128
const N_LM = 468

async function makeTopo(): Promise<FaceTopology> {
  const c = new OffscreenCanvas(512, 512)
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, 512, 512)
  return {
    count: (topoJson as any).count,
    uv: new Float32Array((topoJson as any).uv),
    idx: new Float32Array((topoJson as any).idx),
    weightMask: await createImageBitmap(c),
  }
}

function gradient(): Float32Array {
  const d = new Float32Array(H * W * 4)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4
    d[i] = x / (W - 1); d[i + 1] = y / (H - 1); d[i + 2] = ((x * 7 + y * 13) % 32) / 32
  }
  return d
}

// Landmark tensor: landmark i at its canonical UV (crop coords) — the mesh
// becomes an affine image of the canonical face inside the box.
function canonicalLandmarks(topo: FaceTopology): Float32Array {
  const lm = new Float32Array(956)   // 478 × (x, y); mesh uses only 0..467
  for (let v = 0; v < topo.count; v++) {
    const i = topo.idx[v]
    lm[i * 2] = topo.uv[v * 2]
    lm[i * 2 + 1] = topo.uv[v * 2 + 1]
  }
  return lm
}

// Multi-face (slots: 4): the box tensor is 1×4×4 and the landmark tensor packs
// 4 faces end to end. Landmarks are CROP coords, so every face reuses the same
// canonical set — the box slot is what places it in the frame.
function packed(topo: FaceTopology, n: number): Float32Array {
  const one = canonicalLandmarks(topo)
  const out = new Float32Array(one.length * n)
  for (let i = 0; i < n; i++) out.set(one, i * one.length)
  return out
}

describe.each(BACKENDS)('FaceTouchupStage multi-face ($name)', ({ create }) => {
  it('retouches every occupied slot and leaves empty slots alone', async () => {
    const backend = await create()
    const topo = await makeTopo()
    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956 * 4, packed(topo, 4))
    // Two faces present (slots 0,1), two empty (score 0).
    const faces = [{ cx: 0.25, cy: 0.5, hs: 0.12 }, { cx: 0.75, cy: 0.5, hs: 0.12 }]
    const box = backend.tensor(1, 4, 4, new Float32Array([
      faces[0].cx, faces[0].cy, faces[0].hs, 0.9,
      faces[1].cx, faces[1].cy, faces[1].hs, 0.9,
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]))
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, {
      strength: 1, amount: 16, detail: 0, thresh: 0.15, slots: 4,
    })
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    const diffAt = (x: number, y: number) => {
      const p = (y * W + x) * 4
      return Math.abs(got[p] - img[p]) + Math.abs(got[p + 1] - img[p + 1]) + Math.abs(got[p + 2] - img[p + 2])
    }
    const boxPx = (f: { cx: number; cy: number; hs: number }) => ({
      x0: f.cx * W - f.hs * W, x1: f.cx * W + f.hs * W,
      y0: f.cy * H - f.hs * W, y1: f.cy * H + f.hs * W,
    })
    const b = faces.map(boxPx)

    // Both faces must actually change — the single-face bug is that only slot 0
    // (or one merged box) gets touched.
    const inside = b.map(r => {
      let m = 0
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) m = Math.max(m, diffAt(x, y))
      return m
    })
    expect(inside[0]).toBeGreaterThan(0.02)
    expect(inside[1]).toBeGreaterThan(0.02)

    // Well outside both boxes, nothing moved — in particular the empty slots
    // (which would otherwise smear a degenerate box at the origin).
    let outside = 0
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const near = b.some(r => x >= r.x0 - 4 && x <= r.x1 + 4 && y >= r.y0 - 4 && y <= r.y1 + 4)
      if (!near) outside = Math.max(outside, diffAt(x, y))
    }
    expect(outside).toBeLessThan(1e-5)
  })

  it('setActiveSlots bounds the draw even when a later slot has a live box', async () => {
    // The renderer gates landmark runs on a throttled occupancy probe, so between
    // probes a slot can hold a LIVE box while its landmark tensor is stale. Drawing
    // it would smear the previous face's mesh onto the new one — worse than not
    // retouching. setActiveSlots is what makes the draw follow the landmark runs.
    const backend = await create()
    const topo = await makeTopo()
    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956 * 4, packed(topo, 4))
    const box = backend.tensor(1, 4, 4, new Float32Array([
      0.25, 0.5, 0.12, 0.9,
      0.75, 0.5, 0.12, 0.9,   // live, but beyond activeSlots below
      0, 0, 0, 0,
      0, 0, 0, 0,
    ]))
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, {
      strength: 1, amount: 16, detail: 0, thresh: 0.15, slots: 4,
    })
    stage.setActiveSlots(1)
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    const maxDiffIn = (cx: number) => {
      let m = 0
      const x0 = cx * W - 0.12 * W, x1 = cx * W + 0.12 * W
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (x < x0 || x > x1) continue
        const p = (y * W + x) * 4
        m = Math.max(m, Math.abs(got[p] - img[p]) + Math.abs(got[p + 1] - img[p + 1]) + Math.abs(got[p + 2] - img[p + 2]))
      }
      return m
    }
    expect(maxDiffIn(0.25)).toBeGreaterThan(0.02)   // slot 0 drawn
    expect(maxDiffIn(0.75)).toBeLessThan(1e-5)      // slot 1 gated off despite a live box
  })

  it('samples the weight mask UNTILED at slots:4 (a tiled lookup reads the wrong quadrant)', async () => {
    // Regression. The atlas IS tiled per face, so it's sampled with the tiled uv.
    // The weight mask is NOT — it's one canonical 512² asset shared by every face.
    // Sampling it with the tiled uv made face 0 read only the mask's top-left
    // QUADRANT, stretched over the whole face: invisible at slots:1 (tile_uv is
    // identity) and badly wrong at slots:4, which is exactly what shipped to
    // medium/large/xl.
    //
    // The other tests here can't see it: they use an ALL-WHITE mask, so reading
    // the wrong region returns the same value. This one puts the weight only on
    // the mask's RIGHT half (u > 0.5) — under the bug face 0 samples u∈[0,0.5],
    // gets all-black, and nothing is retouched at all.
    const backend = await create()
    const topo = await makeTopo()
    const half = new OffscreenCanvas(512, 512)
    const hctx = half.getContext('2d')!
    hctx.fillStyle = '#000'; hctx.fillRect(0, 0, 512, 512)
    hctx.fillStyle = '#fff'; hctx.fillRect(256, 0, 256, 512)   // white only where u > 0.5
    const topoHalf: FaceTopology = { ...topo, weightMask: await createImageBitmap(half) }

    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956 * 4, packed(topo, 4))
    const box = backend.tensor(1, 4, 4, new Float32Array([
      0.5, 0.5, 0.15, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]))
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topoHalf, {
      strength: 1, amount: 16, detail: 0, thresh: 0.15, slots: 4,
    })
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    let maxDiff = 0
    for (let p = 0; p < H * W; p++)
      for (let c = 0; c < 3; c++)
        maxDiff = Math.max(maxDiff, Math.abs(got[p * 4 + c] - img[p * 4 + c]))
    // Correct: the mask's white half lands on the face and retouches it.
    // Tiled lookup: face 0 reads only black → zero weight → nothing changes.
    expect(maxDiff).toBeGreaterThan(0.02)
  })

  it('slots:4 with one face ≈ slots:1 with that face (tiling is not a behavior change)', async () => {
    const topo = await makeTopo()
    const img = gradient()
    const face = new Float32Array([0.5, 0.5, 0.15, 0.9])
    const params = { strength: 1, amount: 16, detail: 0, thresh: 0.15 }

    const runWith = async (slots: number) => {
      const backend = await create()
      const frame = backend.tensor(H, W, 4, img)
      const lm = backend.tensor(1, 1, 956 * slots, packed(topo, slots))
      const bx = new Float32Array(slots * 4)
      bx.set(face, 0)
      const box = backend.tensor(1, slots, 4, bx)
      const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, { ...params, slots })
      stage.run()
      const out = await backend.readback(stage.output)
      backend.destroy()
      return out
    }
    const one = await runWith(1)
    const four = await runWith(4)

    // The 2×2 atlas halves each face's tile resolution, so this is a similarity
    // check, not equality: the smoothing must land in the same place with the
    // same character. A regression that broke tile mapping would blow past this.
    let maxErr = 0
    for (let p = 0; p < H * W; p++)
      for (let c = 0; c < 3; c++)
        maxErr = Math.max(maxErr, Math.abs(four[p * 4 + c] - one[p * 4 + c]))
    expect(maxErr).toBeLessThan(0.15)
  })
})

describe.each(BACKENDS)('FaceTouchupStage ($name)', ({ create }) => {
  it('no face (score 0) → exact passthrough', async () => {
    const backend = await create()
    const topo = await makeTopo()
    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956, canonicalLandmarks(topo))
    const box = backend.tensor(1, 1, 4, new Float32Array([0.5, 0.5, 0.2, 0]))   // score 0
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, {
      strength: 1, amount: 8, detail: 0.3, thresh: 0.15,
    })
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    let maxErr = 0
    for (let p = 0; p < H * W; p++)
      for (let c = 0; c < 3; c++)
        maxErr = Math.max(maxErr, Math.abs(got[p * 4 + c] - img[p * 4 + c]))
    expect(maxErr).toBeLessThan(1e-5)
  })

  it('strength 0 with a face → output ≈ input', async () => {
    const backend = await create()
    const topo = await makeTopo()
    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956, canonicalLandmarks(topo))
    const box = backend.tensor(1, 1, 4, new Float32Array([0.5, 0.5, 0.2, 0.9]))
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, {
      strength: 0, amount: 8, detail: 0.3, thresh: 0.15,
    })
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    let maxErr = 0
    for (let p = 0; p < H * W; p++)
      for (let c = 0; c < 3; c++)
        maxErr = Math.max(maxErr, Math.abs(got[p * 4 + c] - img[p * 4 + c]))
    // Mesh redraw resamples the frame bilinearly; the rasterizer's ~1e-3-texel
    // interpolation offset shows up against this test image's high-frequency
    // third channel (0.2–0.8 jumps between adjacent pixels). Identical on both
    // backends — deterministic sampling noise, not smoothing leakage.
    expect(maxErr).toBeLessThan(5e-3)
  })

  it('strength 1 changes the face region and nothing far outside the box', async () => {
    const backend = await create()
    const topo = await makeTopo()
    const img = gradient()
    const frame = backend.tensor(H, W, 4, img)
    const lm = backend.tensor(1, 1, 956, canonicalLandmarks(topo))
    const boxV = { cx: 0.5, cy: 0.5, hs: 0.15, score: 0.9 }
    const box = backend.tensor(1, 1, 4, new Float32Array([boxV.cx, boxV.cy, boxV.hs, boxV.score]))
    const stage = backend.ops.FaceTouchupStage(frame, lm, box, topo, {
      strength: 1, amount: 16, detail: 0, thresh: 0.15,
    })
    stage.run()
    const got = await backend.readback(stage.output)
    backend.destroy()

    // Box extents in px (hs is a fraction of WIDTH; y half-extent aspect-scaled).
    const hsx = boxV.hs * W, hsy = boxV.hs * W
    const x0 = boxV.cx * W - hsx, x1 = boxV.cx * W + hsx
    const y0 = boxV.cy * H - hsy, y1 = boxV.cy * H + hsy
    let insideDiff = 0, outsideDiff = 0
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = (y * W + x) * 4
      const d = Math.abs(got[p] - img[p]) + Math.abs(got[p + 1] - img[p + 1]) + Math.abs(got[p + 2] - img[p + 2])
      const inside = x >= x0 && x <= x1 && y >= y0 && y <= y1
      if (inside) insideDiff = Math.max(insideDiff, d)
      else if (x < x0 - 4 || x > x1 + 4 || y < y0 - 4 || y > y1 + 4) outsideDiff = Math.max(outsideDiff, d)
    }
    expect(insideDiff).toBeGreaterThan(0.02)   // smoothing visibly changed the face
    expect(outsideDiff).toBeLessThan(1e-5)     // strictly untouched outside the box
  })
})
