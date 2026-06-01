import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { ModelWeights, MBConvWeights } from '~/model/weights.ts'
import { DepthwiseSeparable } from '~/model/blocks/depthwise_separable.ts'
import { MBConv } from '~/model/blocks/mbconv.ts'
import { DecoderBlock } from '~/model/blocks/decoder_block.ts'

// XL: 320×180 base, tf_efficientnet_lite4 encoder (deeper than lite0 — stages
// have 4/4/6/6/8 blocks), xl decoder (256,128,64,32). Skips: stage1 (32ch @/4),
// stage2 (56ch @/8), stage4 (160ch @/16); bottleneck-in = stage6 (448ch @/32).
//
// Per-block MBConv spec [in, mid, out, kernel, stride] per stage (from timm
// tf_efficientnet_lite4). expand-ratio 6; the first block of each stage changes
// channels (and strides where shown).
interface BlockSpec { in: number; mid: number; out: number; k: number; s: number }
const STAGES: BlockSpec[][] = [
  // s1 (4)
  [ { in: 24,  mid: 144,  out: 32,  k: 3, s: 2 }, { in: 32,  mid: 192,  out: 32,  k: 3, s: 1 },
    { in: 32,  mid: 192,  out: 32,  k: 3, s: 1 }, { in: 32,  mid: 192,  out: 32,  k: 3, s: 1 } ],
  // s2 (4)
  [ { in: 32,  mid: 192,  out: 56,  k: 5, s: 2 }, { in: 56,  mid: 336,  out: 56,  k: 5, s: 1 },
    { in: 56,  mid: 336,  out: 56,  k: 5, s: 1 }, { in: 56,  mid: 336,  out: 56,  k: 5, s: 1 } ],
  // s3 (6)
  [ { in: 56,  mid: 336,  out: 112, k: 3, s: 2 }, { in: 112, mid: 672,  out: 112, k: 3, s: 1 },
    { in: 112, mid: 672,  out: 112, k: 3, s: 1 }, { in: 112, mid: 672,  out: 112, k: 3, s: 1 },
    { in: 112, mid: 672,  out: 112, k: 3, s: 1 }, { in: 112, mid: 672,  out: 112, k: 3, s: 1 } ],
  // s4 (6)
  [ { in: 112, mid: 672,  out: 160, k: 5, s: 1 }, { in: 160, mid: 960,  out: 160, k: 5, s: 1 },
    { in: 160, mid: 960,  out: 160, k: 5, s: 1 }, { in: 160, mid: 960,  out: 160, k: 5, s: 1 },
    { in: 160, mid: 960,  out: 160, k: 5, s: 1 }, { in: 160, mid: 960,  out: 160, k: 5, s: 1 } ],
  // s5 (8)
  [ { in: 160, mid: 960,  out: 272, k: 5, s: 2 }, { in: 272, mid: 1632, out: 272, k: 5, s: 1 },
    { in: 272, mid: 1632, out: 272, k: 5, s: 1 }, { in: 272, mid: 1632, out: 272, k: 5, s: 1 },
    { in: 272, mid: 1632, out: 272, k: 5, s: 1 }, { in: 272, mid: 1632, out: 272, k: 5, s: 1 },
    { in: 272, mid: 1632, out: 272, k: 5, s: 1 }, { in: 272, mid: 1632, out: 272, k: 5, s: 1 } ],
  // s6 (1)
  [ { in: 272, mid: 1632, out: 448, k: 3, s: 1 } ],
]

export class EfficientNetLiteMattingXL {
  readonly output: Tensor
  // Pre-head feature (finalUp output, at base-input/2 res). A UNet wrapper
  // consumes this (upsampled to input res) instead of `output`.
  readonly featLowRes: Tensor

  private readonly stem: Op
  private readonly s0:   DepthwiseSeparable
  readonly stages: MBConv[][]            // [stageIdx][blockIdx]; exposed for tests
  private readonly bottleneck: Op
  private readonly dec0: DecoderBlock
  private readonly dec1: DecoderBlock
  private readonly dec2: DecoderBlock
  private readonly finalUp: Op
  private readonly outConv: Op
  private readonly alpha:   Op

  constructor(backend: Backend, input: Tensor, w: ModelWeights) {
    this.stem = backend.ops.Conv2d(input, w.encoder.stem, {
      outChannels: 32, kernel: 3, stride: 2, padding: 'same', activation: 'relu6',
    })
    this.s0 = new DepthwiseSeparable(backend, this.stem.output, w.encoder.s0, {
      outChannels: 24, kernel: 3, stride: 1, padding: 1,
    })

    const stageW: MBConvWeights[][] = [
      w.encoder.s1, w.encoder.s2!, w.encoder.s3!, w.encoder.s4!, w.encoder.s5!, w.encoder.s6!,
    ]
    this.stages = []
    let prev: Tensor = this.s0.output
    for (let si = 0; si < STAGES.length; si++) {
      const blocks: MBConv[] = []
      for (let bi = 0; bi < STAGES[si].length; bi++) {
        const sp = STAGES[si][bi]
        const mb = new MBConv(backend, prev, stageW[si][bi], {
          inChannels:  sp.in,
          midChannels: sp.mid,
          outChannels: sp.out,
          kernel:      sp.k,
          stride:      sp.s,
          padding:     sp.s === 2 ? 'same' : (sp.k - 1) / 2,
        })
        blocks.push(mb)
        prev = mb.output
      }
      this.stages.push(blocks)
    }

    const last = (s: number) => this.stages[s][this.stages[s].length - 1].output
    const feat1 = last(0)   // stage1 out, 32ch  @/4
    const feat2 = last(1)   // stage2 out, 56ch  @/8
    const feat3 = last(3)   // stage4 out, 160ch @/16
    const deepest = last(5) // stage6 out, 448ch @/32

    this.bottleneck = backend.ops.Conv2d(deepest, w.bottleneck, {
      outChannels: 256, kernel: 1, stride: 1, padding: 0, activation: 'relu6',
    })

    this.dec0 = new DecoderBlock(backend, this.bottleneck.output, feat3, w.decoder.blocks[0], { outChannels: 128 })
    this.dec1 = new DecoderBlock(backend, this.dec0.output,       feat2, w.decoder.blocks[1], { outChannels: 64 })
    this.dec2 = new DecoderBlock(backend, this.dec1.output,       feat1, w.decoder.blocks[2], { outChannels: 32 })

    this.finalUp = backend.ops.UpsampleConv1x1(this.dec2.output, w.decoder.finalUpsample, {
      outH: this.dec2.output.h * 2, outW: this.dec2.output.w * 2, outChannels: 32, activation: 'relu6',
    })
    this.featLowRes = this.finalUp.output

    this.outConv = backend.ops.Conv2d(this.finalUp.output, w.decoder.outputConv, {
      outChannels: 4, kernel: 1, stride: 1, padding: 0, activation: 'none',
    })
    this.alpha = backend.ops.UpsampleSigmoid(this.outConv.output, {
      outH: this.outConv.output.h * 2, outW: this.outConv.output.w * 2,
    })
    this.output = this.alpha.output
  }

  run(): void {
    this.stem.run()
    this.s0.run()
    for (const stage of this.stages) for (const b of stage) b.run()
    this.bottleneck.run()
    this.dec0.run()
    this.dec1.run()
    this.dec2.run()
    this.finalUp.run()
    this.outConv.run()
    this.alpha.run()
  }
}
