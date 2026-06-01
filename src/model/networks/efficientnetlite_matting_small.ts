import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { ModelWeights } from '~/model/weights.ts'
import { DepthwiseSeparable } from '~/model/blocks/depthwise_separable.ts'
import { MBConv } from '~/model/blocks/mbconv.ts'
import { DecoderBlock } from '~/model/blocks/decoder_block.ts'

// Small encoder (tf_efficientnet_lite0, out_indices=(1,2,3) — stops at stage 4),
// standard decoder trimmed to 2 skips: dec_ch=(128,64,32). Used by small + xs presets.
export class EfficientNetLiteMattingSmall {
  readonly output: Tensor
  // Pre-head feature (finalUp output, at base-input/2 res). A UNet wrapper
  // consumes this (upsampled to input res) instead of `output`.
  readonly featLowRes: Tensor

  // Encoder
  private readonly stem:  Op
  private readonly s0:    DepthwiseSeparable

  // Stage 1 — 16→24ch (skip feat0: s1b1.output, 24ch@stride-4)
  private readonly s1b0:  MBConv
  private readonly s1b1:  MBConv

  // Stage 2 — 24→40ch (skip feat1: s2b1.output, 40ch@stride-8)
  private readonly s2b0:  MBConv
  private readonly s2b1:  MBConv

  // Stage 3 — 40→80ch
  private readonly s3b0:  MBConv
  private readonly s3b1:  MBConv
  private readonly s3b2:  MBConv

  // Stage 4 — 80→112ch (bottleneck input: s4b2.output, 112ch@stride-16)
  private readonly s4b0:  MBConv
  private readonly s4b1:  MBConv
  private readonly s4b2:  MBConv

  // Bottleneck — 112→128ch
  private readonly bottleneck: Op

  // Decoder (only 2 blocks — small encoder has only 2 skip connections)
  private readonly dec0: DecoderBlock  // deep=128ch@stride-16, skip=feat1 40ch@stride-8 → 64ch
  private readonly dec1: DecoderBlock  // deep= 64ch@stride-8,  skip=feat0 24ch@stride-4 → 32ch

  private readonly finalUp: Op  // UpsampleConv1x1: stride-4 → stride-2, 32ch, relu6
  private readonly outConv: Op  // Conv2d 1×1:      stride-2,            4ch, no-act
  private readonly alpha:   Op  // UpsampleSigmoid: stride-2 → stride-1 (full res)

  constructor(backend: Backend, input: Tensor, w: ModelWeights) {
    this.stem = backend.ops.Conv2d(input, w.encoder.stem, {
      outChannels: 32,
      kernel:      3,
      stride:      2,
      padding:     'same',
      activation:  'relu6',
    })

    this.s0 = new DepthwiseSeparable(backend, this.stem.output, w.encoder.s0, {
      outChannels: 16,
      kernel:      3,
      stride:      1,
      padding:     1,
    })

    // Stage 1
    this.s1b0 = new MBConv(backend, this.s0.output, w.encoder.s1[0], {
      inChannels:  16,
      midChannels: 96,
      outChannels: 24,
      kernel:      3,
      stride:      2,
      padding:     'same',
    })
    this.s1b1 = new MBConv(backend, this.s1b0.output, w.encoder.s1[1], {
      inChannels:  24,
      midChannels: 144,
      outChannels: 24,
      kernel:      3,
      stride:      1,
      padding:     1,
    })

    // Stage 2
    this.s2b0 = new MBConv(backend, this.s1b1.output, w.encoder.s2[0], {
      inChannels:  24,
      midChannels: 144,
      outChannels: 40,
      kernel:      5,
      stride:      2,
      padding:     'same',
    })
    this.s2b1 = new MBConv(backend, this.s2b0.output, w.encoder.s2[1], {
      inChannels:  40,
      midChannels: 240,
      outChannels: 40,
      kernel:      5,
      stride:      1,
      padding:     2,
    })

    // Stage 3
    this.s3b0 = new MBConv(backend, this.s2b1.output, w.encoder.s3[0], {
      inChannels:  40,
      midChannels: 240,
      outChannels: 80,
      kernel:      3,
      stride:      2,
      padding:     'same',
    })
    this.s3b1 = new MBConv(backend, this.s3b0.output, w.encoder.s3[1], {
      inChannels:  80,
      midChannels: 480,
      outChannels: 80,
      kernel:      3,
      stride:      1,
      padding:     1,
    })
    this.s3b2 = new MBConv(backend, this.s3b1.output, w.encoder.s3[2], {
      inChannels:  80,
      midChannels: 480,
      outChannels: 80,
      kernel:      3,
      stride:      1,
      padding:     1,
    })

    // Stage 4
    this.s4b0 = new MBConv(backend, this.s3b2.output, w.encoder.s4[0], {
      inChannels:  80,
      midChannels: 480,
      outChannels: 112,
      kernel:      5,
      stride:      1,
      padding:     2,
    })
    this.s4b1 = new MBConv(backend, this.s4b0.output, w.encoder.s4[1], {
      inChannels:  112,
      midChannels: 672,
      outChannels: 112,
      kernel:      5,
      stride:      1,
      padding:     2,
    })
    this.s4b2 = new MBConv(backend, this.s4b1.output, w.encoder.s4[2], {
      inChannels:  112,
      midChannels: 672,
      outChannels: 112,
      kernel:      5,
      stride:      1,
      padding:     2,
    })

    this.bottleneck = backend.ops.Conv2d(this.s4b2.output, w.bottleneck, {
      outChannels: 128,
      kernel:      1,
      stride:      1,
      padding:     0,
      activation:  'relu6',
    })

    this.dec0 = new DecoderBlock(backend, this.bottleneck.output, this.s2b1.output, w.decoder.blocks[0], { outChannels: 64 })
    this.dec1 = new DecoderBlock(backend, this.dec0.output,       this.s1b1.output, w.decoder.blocks[1], { outChannels: 32 })

    this.finalUp = backend.ops.UpsampleConv1x1(this.dec1.output, w.decoder.finalUpsample, {
      outH:        this.dec1.output.h * 2,
      outW:        this.dec1.output.w * 2,
      outChannels: 32,
      activation:  'relu6',
    })
    this.featLowRes = this.finalUp.output

    this.outConv = backend.ops.Conv2d(this.finalUp.output, w.decoder.outputConv, {
      outChannels: 4,
      kernel:      1,
      stride:      1,
      padding:     0,
      activation:  'none',
    })

    this.alpha = backend.ops.UpsampleSigmoid(this.outConv.output, {
      outH: this.outConv.output.h * 2,
      outW: this.outConv.output.w * 2,
    })

    this.output = this.alpha.output
  }

  run(): void {
    this.stem.run()
    this.s0.run()
    this.s1b0.run()
    this.s1b1.run()
    this.s2b0.run()
    this.s2b1.run()
    this.s3b0.run()
    this.s3b1.run()
    this.s3b2.run()
    this.s4b0.run()
    this.s4b1.run()
    this.s4b2.run()
    this.bottleneck.run()
    this.dec0.run()
    this.dec1.run()
    this.finalUp.run()
    this.outConv.run()
    this.alpha.run()
  }
}
