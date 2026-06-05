import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { FlowWeights } from '~/model/weights.ts'

// Per-stage kernel sizes / pads of the learned contracting path — mirrors training
// FlowEncoderNet (ks = [5,5,3,3][:nTaps], pad = (k-1)/2).
const STAGE_K = [5, 5, 3, 3]
const STAGE_P = [2, 2, 1, 1]

// Resize `t` to ref's resolution if they differ. The learned-path stages vs the
// encoder taps (and the decoder's deconv/upflow vs the skip) can disagree by a
// pixel from padding/rounding — training uses nearest `_match` / `crop_like`; we
// bilinear-match (a base/4-and-below, low-frequency field, so the difference is
// negligible — quantified by the fidelity test).
function matchSize(backend: Backend, t: Tensor, ref: Tensor, steps: Op[]): Tensor {
  if (t.h === ref.h && t.w === ref.w) return t
  const r = backend.ops.BilinearUpsample(t, { outH: ref.h, outW: ref.w })
  steps.push(r)
  return r.output
}

// Optical-flow head: a thin learned contracting path on the base-res RGB frame
// pair, fused with the matting encoder's cached taps at each scale, then a small
// decoder regresses base/4 flow (vector in .xy). Composition mirrors training
// WrapperFlowNet + FlowEncoderNet for the shipping config (base/4, dense, leaky,
// no tap-half). It owns no encoder — `taps` come from TierModel.encoderTaps,
// computed on the previous full frame and read here for free.
//
//   frameA, frameB : base-res RGB (4-ch each; concat → 8-ch, 6 real + 2 dead)
//   taps           : encoder pyramid, finest→coarsest
//   output         : base/4 flow, vector in .xy
export class OpticalFlowNet implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  private readonly steps: Op[]

  constructor(
    backend: Backend, frameA: Tensor, frameB: Tensor, taps: Tensor[],
    w: FlowWeights, decW = 16,
  ) {
    this.inputs = [frameA, frameB, ...taps]
    const steps: Op[] = []

    // Stem: 8-ch (6 real) frame pair → /2.
    const pair = backend.ops.ChannelConcat(frameA, frameB)
    steps.push(pair)
    const stem = backend.ops.Conv2d(pair.output, w.stem, {
      outChannels: decW, kernel: 7, stride: 2, padding: 3, activation: 'leaky',
    })
    steps.push(stem)

    // Contracting stages, each fused (concat) with its encoder tap.
    const fused: Tensor[] = []
    let s = stem.output
    for (let i = 0; i < taps.length; i++) {
      const stage = backend.ops.Conv2d(s, w.stages[i], {
        outChannels: decW, kernel: STAGE_K[i], stride: 2, padding: STAGE_P[i], activation: 'leaky',
      })
      steps.push(stage)
      const tap = matchSize(backend, taps[i], stage.output, steps)
      const cat = backend.ops.ChannelConcat(stage.output, tap)
      steps.push(cat)
      fused.push(cat.output)
      s = cat.output
    }

    // Decoder: bottleneck flow at the coarsest level, refined up to base/4.
    const predictBot = backend.ops.Conv2d(fused[fused.length - 1], w.predictBot, {
      outChannels: 4, kernel: 3, stride: 1, padding: 1, activation: 'none',
    })
    steps.push(predictBot)
    let flow = predictBot.output
    let dec = fused[fused.length - 1]

    for (let j = 0; j < fused.length - 1; j++) {
      const i = fused.length - 2 - j
      const deconv = backend.ops.ConvTranspose2d(dec, w.deconv[j], {
        outChannels: decW, kernel: 4, stride: 2, padding: 1, activation: 'leaky',
      })
      const upflow = backend.ops.ConvTranspose2d(flow, w.upflow[j], {
        outChannels: 4, kernel: 4, stride: 2, padding: 1, activation: 'none',
      })
      steps.push(deconv, upflow)
      const up  = matchSize(backend, deconv.output, fused[i], steps)
      const fup = matchSize(backend, upflow.output, fused[i], steps)
      const cat1 = backend.ops.ChannelConcat(fused[i], up)
      const cat2 = backend.ops.ChannelConcat(cat1.output, fup)
      steps.push(cat1, cat2)
      const pred = backend.ops.Conv2d(cat2.output, w.predict[j], {
        outChannels: 4, kernel: 3, stride: 1, padding: 1, activation: 'none',
      })
      steps.push(pred)
      flow = pred.output
      dec = cat2.output
    }

    this.output = flow   // base/4 flow, vector in .xy
    this.steps = steps
  }

  run(): void {
    for (const op of this.steps) op.run()
  }
}
