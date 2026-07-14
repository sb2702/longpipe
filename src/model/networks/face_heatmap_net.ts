import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { FaceWeights } from '~/model/weights.ts'
import { DecoderBlock } from '~/model/blocks/decoder_block.ts'

// Face-keypoint heatmap decoder — mirrors training MattingModel._face_decode.
// A separate decoder on the shared matting encoder: it owns no encoder and
// consumes `taps` from the base network (TierModel.encoderTaps), computed for
// free during the matting pass. Input = the coarsest tap; the two next-finer
// taps are DecoderBlock skips, so the output lands two strides up from the
// encoder's deepest level: base/8 for the full encoders (deepest /32), base/4
// for the small encoder (deepest /16) — the stride falls out of encoder depth,
// no per-tier config.
//
// Output: post-SIGMOID heatmaps, channels 0-4 = L-eye, R-eye, nose, L-mouth,
// R-mouth (RetinaFace order); channels 5-7 dead (sigmoid(0) = 0.5 — consumers
// must read ch < 5). Downstream decode (face box / landmark crop) must use a
// windowed soft-argmax centroid around each channel's peak, NOT hard argmax —
// the grid is coarse (e.g. 32×20 at xs) and whole-cell snapping jitters the
// crop (see training/eval/visualize_live_webcam.py detect_face_box).
const PROJ_C = 64
const BLOCK1_C = 48
const BLOCK2_C = 32
const OUT_C = 8   // 5 real heatmaps, padded to vec4

export class FaceHeatmapNet implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  private readonly steps: Op[]

  // taps: the base network's encoderTaps (finest→coarsest, length ≥ 3).
  constructor(backend: Backend, taps: Tensor[], w: FaceWeights) {
    if (taps.length < 3)
      throw new Error(`FaceHeatmapNet: needs ≥3 encoder taps, got ${taps.length}`)
    const deepest = taps[taps.length - 1]
    const skip1   = taps[taps.length - 2]
    const skip2   = taps[taps.length - 3]
    this.inputs = [deepest, skip1, skip2]
    const steps: Op[] = []

    const proj = backend.ops.Conv2d(deepest, w.proj, {
      outChannels: PROJ_C, kernel: 1, stride: 1, padding: 0, activation: 'relu6',
    })
    steps.push(proj)

    const b1 = new DecoderBlock(backend, proj.output, skip1, w.block1, { outChannels: BLOCK1_C })
    const b2 = new DecoderBlock(backend, b1.output,   skip2, w.block2, { outChannels: BLOCK2_C })
    steps.push(b1, b2)

    const logits = backend.ops.Conv2d(b2.output, w.out, {
      outChannels: OUT_C, kernel: 1, stride: 1, padding: 0, activation: 'none',
    })
    const heatmaps = backend.ops.Sigmoid(logits.output)
    steps.push(logits, heatmaps)

    this.output = heatmaps.output
    this.steps = steps
  }

  run(): void {
    for (const op of this.steps) op.run()
  }
}
