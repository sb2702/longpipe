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
