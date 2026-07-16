import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'

// ReframeState + Reframe — the auto-reframe camera. Driven by synthetic box
// tensors: what's under test is the rule (gravity pull, containment, the corner
// no-op, deadband/ease, the manual modes), not the face decode that feeds it.

const ASPECT = 640 / 400
const P = { zoom: 1.35, gravity: 0.5, margin: 0.04, deadband: 0.09, ease: 0.07, aspect: ASPECT }
const AUTO = 0, HOLD = 1, SOLVE = 2

const boxes = (b: number[][], k = 4) => {
  const a = new Float32Array(k * 4)
  b.forEach((v, i) => a.set(v, i * 4))
  return a
}

async function step(create: () => Promise<any>, boxData: Float32Array, prev: number[], mode: number, params = P) {
  const backend = await create()
  const k = boxData.length / 4
  const bt = backend.tensor(1, k, 4, boxData)
  const pt = backend.tensor(1, 1, 4, new Float32Array(prev))
  const ct = backend.tensor(1, 1, 4, new Float32Array([mode, 0, 0, 0]))
  const op = backend.ops.ReframeState(bt, pt, ct, params)
  op.run()
  const out = await backend.readback(op.output)
  backend.destroy()
  return { cx: out[0], cy: out[1], size: out[2], moving: out[3] }
}

describe.each(BACKENDS)('ReframeState ($name)', ({ create }) => {
  it('pulls toward the subject without centring it (gravity 0.5)', async () => {
    // Face at x=0.8; gravity 0.5 wants centre 0.65. At zoom 1.35 the crop is
    // 0.74 wide, so the frame edge clamps the centre to 1 - 0.37 = 0.63.
    const s = await step(create, boxes([[0.8, 0.5, 0.08, 0.9]]), [0, 0, 0, 0], AUTO)
    expect(s.cx).toBeGreaterThan(0.5)    // moved toward the subject
    expect(s.cx).toBeLessThan(0.8)       // but did NOT centre it
    expect(s.size).toBeGreaterThan(0)
  })

  it('a centred subject leaves the crop centred', async () => {
    const s = await step(create, boxes([[0.5, 0.5, 0.08, 0.9]]), [0, 0, 0, 0], AUTO)
    expect(s.cx).toBeCloseTo(0.5, 3)
    expect(s.cy).toBeCloseTo(0.5, 3)
    expect(s.size).toBeCloseTo(1 / 1.35, 2)   // full requested zoom — nothing binds
  })

  it('a subject in the corner falls back to the full frame — the no-op', async () => {
    // The behaviour Meet shows: head hard into the corner, reframe does nothing.
    // No special case — once the centre clamps to the frame edge, containment is
    // x >= half+margin at ANY zoom, so it never fits and the solve falls through
    // to its full-frame fallback.
    const s = await step(create, boxes([[0.06, 0.06, 0.05, 0.9]]), [0, 0, 0, 0], AUTO)
    expect(s.size).toBeCloseTo(1, 3)
    expect(s.cx).toBeCloseTo(0.5, 3)
  })

  it('backs the zoom off for a subject too big for the requested crop', async () => {
    // What the relaxation is actually for. hs 0.31 → the face is wider than the
    // 1.35 crop allows, so zoom relaxes until it fits (but not all the way to 1).
    const s = await step(create, boxes([[0.5, 0.5, 0.31, 0.9]]), [0, 0, 0, 0], AUTO)
    expect(s.size).toBeGreaterThan(1 / 1.35)   // wider than requested
    expect(s.size).toBeLessThanOrEqual(1)
  })

  it('picks the largest face as the subject', async () => {
    const s = await step(create, boxes([
      [0.2, 0.5, 0.04, 0.9],   // small, left
      [0.8, 0.5, 0.10, 0.9],   // large, right → should win
    ]), [0, 0, 0, 0], AUTO)
    expect(s.cx).toBeGreaterThan(0.5)
  })

  it('no face → holds the previous frame', async () => {
    const prev = [0.42, 0.47, 0.74, 0]
    const s = await step(create, boxes([]), prev, AUTO)
    expect(s.cx).toBeCloseTo(prev[0], 5)
    expect(s.cy).toBeCloseTo(prev[1], 5)
    expect(s.size).toBeCloseTo(prev[2], 5)
  })

  it('deadband holds against small motion and releases on large', async () => {
    const prev = [0.5, 0.5, 1 / 1.35, 0]
    // Target barely moves → inside the deadband → frozen.
    const small = await step(create, boxes([[0.52, 0.5, 0.08, 0.9]]), prev, AUTO)
    expect(small.cx).toBeCloseTo(prev[0], 4)
    expect(small.moving).toBe(0)
    // Target jumps → deadband breached → eases (partway, not a snap).
    const big = await step(create, boxes([[0.75, 0.5, 0.06, 0.9]]), prev, AUTO)
    expect(big.moving).toBe(1)
    expect(big.cx).toBeGreaterThan(prev[0])
    expect(big.cx).toBeLessThan(0.56)    // eased, nowhere near the target yet
  })

  it('manual: hold freezes, solve snaps', async () => {
    const prev = [0.5, 0.5, 1 / 1.35, 0]
    const subject = boxes([[0.75, 0.5, 0.06, 0.9]])
    const held = await step(create, subject, prev, HOLD)
    expect(held.cx).toBeCloseTo(prev[0], 5)
    expect(held.size).toBeCloseTo(prev[2], 5)

    const solved = await step(create, subject, prev, SOLVE)
    expect(solved.cx).toBeCloseTo(0.625, 2)   // snapped straight to the target
    expect(solved.moving).toBe(0)
  })

  it('uninitialised state snaps rather than easing in from nothing', async () => {
    // Also what makes manual "reframe once when enabled, then freeze".
    const s = await step(create, boxes([[0.5, 0.5, 0.08, 0.9]]), [0, 0, 0, 0], HOLD)
    expect(s.size).toBeCloseTo(1 / 1.35, 2)
  })

  it('gravity 0 never moves the centre', async () => {
    const s = await step(create, boxes([[0.85, 0.5, 0.05, 0.9]]), [0, 0, 0, 0], AUTO,
      { ...P, gravity: 0 })
    expect(s.cx).toBeCloseTo(0.5, 3)
  })
})

describe.each(BACKENDS)('Reframe ($name)', ({ create }) => {
  const H = 16, W = 24
  const gradient = () => {
    const d = new Float32Array(H * W * 4)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4
      d[i] = x / (W - 1); d[i + 1] = y / (H - 1); d[i + 2] = 0.25; d[i + 3] = 1
    }
    return d
  }

  it('an uninitialised rect is a bit-exact identity', async () => {
    // Matters: the op is wired in before any face exists, so it must cost nothing
    // visually until the camera has solved.
    const backend = await create()
    const img = gradient()
    const src = backend.tensor(H, W, 4, img)
    const rect = backend.tensor(1, 1, 4, new Float32Array([0, 0, 0, 0]))
    const op = backend.ops.Reframe(src, rect)
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()
    let maxErr = 0
    for (let i = 0; i < H * W * 4; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - img[i]))
    expect(maxErr).toBeLessThan(1e-6)
  })

  it('a full-frame rect (size 1, centred) is also identity', async () => {
    const backend = await create()
    const img = gradient()
    const src = backend.tensor(H, W, 4, img)
    const rect = backend.tensor(1, 1, 4, new Float32Array([0.5, 0.5, 1, 0]))
    const op = backend.ops.Reframe(src, rect)
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()
    let maxErr = 0
    for (let i = 0; i < H * W * 4; i++) maxErr = Math.max(maxErr, Math.abs(got[i] - img[i]))
    expect(maxErr).toBeLessThan(1e-6)
  })

  it('a half-size centred rect magnifies about the centre', async () => {
    const backend = await create()
    const src = backend.tensor(H, W, 4, gradient())
    const rect = backend.tensor(1, 1, 4, new Float32Array([0.5, 0.5, 0.5, 0]))
    const op = backend.ops.Reframe(src, rect)
    op.run()
    const got = await backend.readback(op.output)
    backend.destroy()
    // Output x runs over source x ∈ [0.25, 0.75] of the frame → red channel
    // (which is just x/(W-1)) should span roughly that range, not [0, 1].
    expect(got[0]).toBeCloseTo(0.25, 1)                       // top-left
    expect(got[((H - 1) * W + (W - 1)) * 4]).toBeCloseTo(0.75, 1)   // bottom-right
  })
})
