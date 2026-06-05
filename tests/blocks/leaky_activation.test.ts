import { describe, it, expect } from 'vitest'
import { BACKENDS } from '../helpers/backends'
import conv2d_3x3 from '../fixtures/conv2d_3x3.json'

// LeakyReLU(0.1) was added to the Conv2d activation path for the optical-flow
// net, whose conv stages are leaky-trained. No new PyTorch fixture: the test is
// self-contained — Conv2d('leaky') must equal leaky(Conv2d('none')) elementwise,
// and the conv must actually produce negatives (so the new branch is exercised).
const fx = conv2d_3x3 as {
  kernel_size: number; stride: number; padding: number; out_channels: number
  input_shape: [number, number, number, number]
  input: number[]; weights: number[]; bias: number[]
}
const THRESHOLD = 1e-5

describe.each(BACKENDS)('Conv2d leaky activation ($name)', ({ create }) => {
  it('equals leaky(conv-none) and exercises the negative branch', async () => {
    const backend = await create()
    const [, C, H, W] = fx.input_shape
    const input = backend.tensor(H, W, C, new Float32Array(fx.input))
    const conv = (activation: 'none' | 'leaky') => backend.ops.Conv2d(
      input, { weights: fx.weights, bias: fx.bias },
      { outChannels: fx.out_channels, kernel: fx.kernel_size, stride: fx.stride,
        padding: fx.padding, activation },
    )

    const none = conv('none'); none.run()
    const pre = (await backend.readback(none.output)).slice()
    const leaky = conv('leaky'); leaky.run()
    const got = await backend.readback(leaky.output)
    backend.destroy()

    let maxErr = 0, negatives = 0
    for (let i = 0; i < pre.length; i++) {
      const exp = pre[i] >= 0 ? pre[i] : 0.1 * pre[i]
      maxErr = Math.max(maxErr, Math.abs(got[i] - exp))
      if (pre[i] < 0) negatives++
    }

    expect(negatives).toBeGreaterThan(0)        // the conv produced negatives → branch tested
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
