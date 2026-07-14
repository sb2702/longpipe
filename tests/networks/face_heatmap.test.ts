import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import face_small from '../fixtures/face_small.json'
import face_xs from '../fixtures/face_xs.json'

// Fidelity: the SDK FaceHeatmapNet vs the real trained face head on the same
// packed weights + encoder taps (training/deploy/gen_face_fixture.py — input is
// a real face frame, so the heatmaps have confident peaks, not just background).
// Taps ship in the fixture: this isolates the face decoder; the encoder is
// validated elsewhere.
const FIXTURES = [
  { name: 'small', fx: face_small as any },   // small encoder → base/4 (48×28)
  { name: 'xs',    fx: face_xs    as any },   // small encoder → base/4 (32×20)
]
const N_KP = 5
const WIN = 3   // soft-argmax window, ±cells around the peak (matches training PoC)

// Windowed soft-argmax centroid — the REQUIRED decode for these heatmaps (the
// grid is coarse; hard argmax snaps to whole cells and jitters the face crop).
// Mirrors training/eval/visualize_live_webcam.py detect_face_box. `stride` is
// the channel count of the flat NHWC buffer (8 for SDK output, 5 for the
// PyTorch reference).
function softArgmax(hm: ArrayLike<number>, h: number, w: number, k: number, stride: number) {
  let peak = -Infinity, pr = 0, pc = 0
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const v = hm[(r * w + c) * stride + k]
    if (v > peak) { peak = v; pr = r; pc = c }
  }
  const r0 = Math.max(0, pr - WIN), r1 = Math.min(h, pr + WIN + 1)
  const c0 = Math.max(0, pc - WIN), c1 = Math.min(w, pc + WIN + 1)
  let wsum = 0, ry = 0, cx = 0
  for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
    const v = hm[(r * w + c) * stride + k]
    wsum += v; ry += v * r; cx += v * c
  }
  wsum = Math.max(wsum, 1e-6)
  return { x: cx / wsum, y: ry / wsum, score: peak }   // continuous cell coords
}

describe.each(BACKENDS)('FaceHeatmapNet ($name)', ({ name: backendName, create }) => {
  for (const { name, fx } of FIXTURES) {
    it(`${name}: heatmaps match the PyTorch reference`, async () => {
      const backend = await create()
      const taps = fx.taps.map((t: number[], i: number) => {
        const [h, w, c] = fx.tapShapes[i]
        return backend.tensor(h, w, c, new Float32Array(t))
      })
      const net = new FaceHeatmapNet(backend, taps, fx.faceWeights)
      net.run()
      const got = await backend.readback(net.output)
      backend.destroy()

      const [eh, ew] = fx.expectedHW
      expect(net.output.h).toBe(eh)
      expect(net.output.w).toBe(ew)
      expect(net.output.c).toBe(8)

      // Per-value fidelity over the 5 real channels (SDK buffer is 8-ch NHWC).
      let maxErr = 0
      for (let p = 0; p < eh * ew; p++) {
        for (let k = 0; k < N_KP; k++) {
          const d = Math.abs(got[p * 8 + k] - fx.expected[p * N_KP + k])
          if (d > maxErr) maxErr = d
        }
      }

      // Sub-pixel decode agreement: soft-argmax keypoints from the SDK heatmaps
      // vs from the reference must land within a small fraction of a cell.
      let maxKpErr = 0
      for (let k = 0; k < N_KP; k++) {
        const a = softArgmax(got, eh, ew, k, 8)
        const b = softArgmax(fx.expected, eh, ew, k, N_KP)
        expect(b.score, `${name} kp${k} reference peak`).toBeGreaterThan(0.3)
        maxKpErr = Math.max(maxKpErr, Math.abs(a.x - b.x), Math.abs(a.y - b.y))
      }
      console.log(`[${backendName}] face_${name}: maxErr=${maxErr.toExponential(2)} maxKpErr=${maxKpErr.toExponential(2)} cells`)

      expect(maxErr).toBeLessThan(5e-3)
      expect(maxKpErr).toBeLessThan(0.05)
    })
  }
})
