import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { ConvGRUWeights } from '~/model/weights.ts'

export interface ConvGRUParams {
  // Channel split: passthrough channels pass through unchanged; recurrent
  // channels go through the GRU. passthrough + recurrent must equal the input
  // tensor's channel count when both halves are concatenated for `.output`.
  passthrough: number   // can be 0 when split_ratio=1.0 was used in training
  recurrent:   number
  kernel?:     number   // defaults to 3
}

// Mirrors training/models/temporal_model.py:ConvGRU.forward. The gates conv is
// pre-split into z_conv + r_conv at export time (see ConvGRUWeights doc).
//
// Inputs:
//   a       : passthrough slice  [B, passthrough, H, W]   (may be null if passthrough=0)
//   b       : recurrent slice    [B, recurrent,   H, W]
//   h_prev  : previous hidden    [B, recurrent,   H, W]   (zero buffer on frame 0)
//
// Outputs:
//   .output       : concat([a, b_out])  [B, passthrough + recurrent, H, W]
//   .hiddenState  : h_new               [B, recurrent, H, W]
//
// The runtime threads `.hiddenState` from frame t into `h_prev` of frame t+1.
export class ConvGRU implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  readonly hiddenState: Tensor

  // Exposed for debug/verification — each intermediate tensor in the chain.
  readonly intermediates: Record<string, Tensor>

  private readonly steps: Op[]

  constructor(
    backend: Backend,
    a: Tensor | null,
    b: Tensor,
    h_prev: Tensor,
    w: ConvGRUWeights,
    params: ConvGRUParams,
  ) {
    const c = params.recurrent
    const k = params.kernel ?? 3
    const pad = k >> 1

    this.inputs = a ? [a, b, h_prev] : [b, h_prev]

    // 1. concat([b, h_prev]) → 2c
    const cat_bh = backend.ops.ChannelConcat(b, h_prev)

    // 2. z = sigmoid(z_conv(cat_bh))
    const z_pre = backend.ops.Conv2d(cat_bh.output, w.zConv, {
      outChannels: c, kernel: k, stride: 1, padding: pad, activation: 'none',
    })
    const z = backend.ops.Sigmoid(z_pre.output)

    // 3. r = sigmoid(r_conv(cat_bh))
    const r_pre = backend.ops.Conv2d(cat_bh.output, w.rConv, {
      outChannels: c, kernel: k, stride: 1, padding: pad, activation: 'none',
    })
    const r = backend.ops.Sigmoid(r_pre.output)

    // 4. rh = r ⊙ h_prev
    const rh = backend.ops.ElementwiseMul(r.output, h_prev)

    // 5. h_til = tanh(cand(concat([b, rh])))
    const cat_brh = backend.ops.ChannelConcat(b, rh.output)
    const cand_pre = backend.ops.Conv2d(cat_brh.output, w.cand, {
      outChannels: c, kernel: k, stride: 1, padding: pad, activation: 'none',
    })
    const h_til = backend.ops.Tanh(cand_pre.output)

    // 6. h_new = (1 - z) ⊙ h_prev + z ⊙ h_til
    const h_new = backend.ops.GruUpdate(z.output, h_prev, h_til.output)
    this.hiddenState = h_new.output

    // 7. b_out = b + γ ⊙ h_new
    const b_out = backend.ops.GammaResidual(b, h_new.output, w.gamma)

    // 8. output = concat([a, b_out])  — skip the concat when passthrough = 0.
    if (a) {
      const out = backend.ops.ChannelConcat(a, b_out.output)
      this.output = out.output
      this.steps = [cat_bh, z_pre, z, r_pre, r, rh, cat_brh, cand_pre, h_til,
                    h_new, b_out, out]
    } else {
      this.output = b_out.output
      this.steps = [cat_bh, z_pre, z, r_pre, r, rh, cat_brh, cand_pre, h_til,
                    h_new, b_out]
    }

    this.intermediates = {
      cat_bh:    cat_bh.output,
      z_pre:     z_pre.output,
      z:         z.output,
      r_pre:     r_pre.output,
      r:         r.output,
      rh:        rh.output,
      cat_brh:   cat_brh.output,
      cand_pre:  cand_pre.output,
      h_til:     h_til.output,
      h_new:     h_new.output,
      b_out:     b_out.output,
    }
  }

  run(): void {
    for (const op of this.steps) op.run()
  }
}
