import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { LandmarkWeights } from '~/model/weights.ts'

// Face-landmark regressor — mirrors training/landmarks/landmark_model.py
// LandmarkNet (mesh preset: 478 pts × xy, 256 crop, width 0.5, head_c 32,
// dense/k3 backbone chosen over MBConv on WebGPU bandwidth grounds).
//
// stem k3 s2 → 10 dense k3 conv blocks (stride pattern below) → 1×1 head
// bottleneck → fc. The fc (512→956) runs as a kernel-N VALID conv over the
// N×N×32 head activation (N = crop/64, 4 for the 256 crop): a full-spatial
// conv with no padding is exactly the flatten→FC, keeping the whole net on
// the standard Conv2d op with PyTorch's native weight order.
//
// Output: 1×1×956 tensor — 478 (x, y) pairs, normalized to [0,1] of the crop,
// vec4-packed so lm[i] = (out[2i], out[2i+1]).
const STEM_C = 8
const BLOCKS: Array<{ c: number; stride: number }> = [
  { c: 12, stride: 2 }, { c: 12, stride: 1 },   // 128 → 64
  { c: 24, stride: 2 }, { c: 24, stride: 1 },   //  64 → 32
  { c: 48, stride: 2 }, { c: 48, stride: 1 }, { c: 48, stride: 1 },   // 32 → 16
  { c: 64, stride: 2 }, { c: 64, stride: 1 },   //  16 → 8
  { c: 96, stride: 2 },                          //   8 → 4 (extra stage)
]
const HEAD_C = 32
const NUM_PTS = 478
const OUT_C = NUM_PTS * 2   // 956, a multiple of 4

export class LandmarkNet implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  private readonly steps: Op[]

  // input: (crop, crop, 4) — RGB in .xyz (ImageNet-normalized), .w dead.
  constructor(backend: Backend, input: Tensor, w: LandmarkWeights) {
    if (w.blocks.length !== BLOCKS.length)
      throw new Error(`LandmarkNet: expected ${BLOCKS.length} block weights, got ${w.blocks.length}`)
    this.inputs = [input]
    const steps: Op[] = []

    const stem = backend.ops.Conv2d(input, w.stem, {
      outChannels: STEM_C, kernel: 3, stride: 2, padding: 1, activation: 'relu6',
    })
    steps.push(stem)

    let x = stem.output
    for (let i = 0; i < BLOCKS.length; i++) {
      const conv = backend.ops.Conv2d(x, w.blocks[i], {
        outChannels: BLOCKS[i].c, kernel: 3, stride: BLOCKS[i].stride, padding: 1, activation: 'relu6',
      })
      steps.push(conv)
      x = conv.output
    }

    const head = backend.ops.Conv2d(x, w.headConv, {
      outChannels: HEAD_C, kernel: 1, stride: 1, padding: 0, activation: 'relu6',
    })
    steps.push(head)

    // fc-as-conv: kernel spans the full head activation (h == w here), valid
    // padding → 1×1 spatial out.
    const fc = backend.ops.Conv2d(head.output, w.fc, {
      outChannels: OUT_C, kernel: head.output.h, stride: 1, padding: 0, activation: 'none',
    })
    steps.push(fc)

    this.output = fc.output
    this.steps = steps
  }

  run(): void {
    for (const op of this.steps) op.run()
  }
}
