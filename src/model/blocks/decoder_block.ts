import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { DecoderBlockWeights } from '~/model/weights.ts'

export interface DecoderBlockParams {
  outChannels: number
}

export class DecoderBlock implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  private readonly upOp:         Op
  private readonly concatConvOp: Op
  private readonly conv2Op:      Op

  constructor(backend: Backend, deep: Tensor, skip: Tensor, w: DecoderBlockWeights, params: DecoderBlockParams) {
    this.inputs = [deep, skip]

    // Boundary per the tier benchmark: upsample stays a separate dispatch
    // (clean intermediate), then concat(deep_up, skip) folds into conv1.
    this.upOp = backend.ops.BilinearUpsample(deep, { outH: skip.h, outW: skip.w })

    // Fused concat + conv1 (3×3, relu6). Weight in-channels ordered [deep, skip].
    this.concatConvOp = backend.ops.ConcatConv2d(this.upOp.output, skip, w.conv1, {
      outChannels: params.outChannels,
    })

    this.conv2Op = backend.ops.Conv2d(this.concatConvOp.output, w.conv2, {
      outChannels: params.outChannels,
      kernel:      3,
      stride:      1,
      padding:     1,
      activation:  'relu6',
    })

    this.output = this.conv2Op.output
  }

  run(): void {
    this.upOp.run()
    this.concatConvOp.run()
    this.conv2Op.run()
  }
}
