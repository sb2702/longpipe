import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import { pickMediaPipeSelfieVariant } from '~/model/networks/mediapipe_selfie'

// Ops added for the MediaPipe selfie segmenter, checked against plain-JS
// references (no generated fixtures needed): GlobalAvgPool, ChannelScale, and
// the hswish / sigmoid activations on Conv2d + DepthwiseConv2d.

const rand = (n: number, seed = 1) => {
  let s = seed
  return Float32Array.from({ length: n }, () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648) * 8 - 4)
}
const maxErr = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0
  for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}
const hswish  = (v: number) => v * Math.min(Math.max(v + 3, 0), 6) / 6
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

describe('pickMediaPipeSelfieVariant', () => {
  it('16:9 → landscape; 1:1, 4:3, portrait → square', () => {
    expect(pickMediaPipeSelfieVariant(1280, 720)).toBe('landscape')
    expect(pickMediaPipeSelfieVariant(640, 360)).toBe('landscape')
    expect(pickMediaPipeSelfieVariant(512, 512)).toBe('square')
    expect(pickMediaPipeSelfieVariant(640, 480)).toBe('square')
    expect(pickMediaPipeSelfieVariant(720, 1280)).toBe('square')
  })
})

describe.each(BACKENDS)('MediaPipe ops ($name)', ({ create }) => {
  it('GlobalAvgPool = per-channel mean', async () => {
    const backend = await create()
    const [H, W, C] = [37, 53, 12]
    const data = rand(H * W * C, 7)
    const op = backend.ops.GlobalAvgPool(backend.tensor(H, W, C, data))
    op.run()
    const out = await backend.readback(op.output)
    const ref = new Float32Array(C)
    for (let p = 0; p < H * W; p++) for (let c = 0; c < C; c++) ref[c] += data[p * C + c] / (H * W)
    backend.destroy()
    expect(op.output.h * op.output.w).toBe(1)
    expect(maxErr(out, ref)).toBeLessThan(1e-4)
  })

  it('ChannelScale broadcasts a 1×1×C scale', async () => {
    const backend = await create()
    const [H, W, C] = [9, 11, 8]
    const x = rand(H * W * C, 3), s = rand(C, 5)
    const op = backend.ops.ChannelScale(backend.tensor(H, W, C, x), backend.tensor(1, 1, C, s))
    op.run()
    const out = await backend.readback(op.output)
    const ref = x.map((v, i) => v * s[i % C])
    backend.destroy()
    expect(maxErr(out, ref)).toBeLessThan(1e-5)
  })

  it.each(['hswish', 'sigmoid'] as const)('Conv2d + DepthwiseConv2d %s activation', async (activation) => {
    const backend = await create()
    const [H, W, C] = [5, 6, 4]
    const x = rand(H * W * C, 11)
    const f = activation === 'hswish' ? hswish : sigmoid
    // 1×1 conv with identity weights → out = act(x + bias)
    const eye = new Float32Array(16); for (let i = 0; i < 4; i++) eye[i * 4 + i] = 1
    const bias = Float32Array.from([0.5, -0.5, 1, -1])
    const t = backend.tensor(H, W, C, x)
    const conv = backend.ops.Conv2d(t, { weights: eye, bias }, { outChannels: 4, kernel: 1, stride: 1, padding: 'same', activation })
    // 1×1 depthwise with unit weights → same thing
    const dw = backend.ops.DepthwiseConv2d(t, { weights: new Float32Array([1, 1, 1, 1]), bias }, { kernel: 1, stride: 1, padding: 'same', activation })
    conv.run(); dw.run()
    const [a, b] = [await backend.readback(conv.output), await backend.readback(dw.output)]
    const ref = x.map((v, i) => f(v + bias[i % 4]))
    backend.destroy()
    expect(maxErr(a, ref)).toBeLessThan(1e-5)
    expect(maxErr(b, ref)).toBeLessThan(1e-5)
  })
})
