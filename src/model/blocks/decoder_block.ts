import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { DecoderBlockWeights } from '~/model/weights.ts'

export interface DecoderBlockParams {
  outChannels: number
}

export class DecoderBlock implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  private readonly upConcatOp: Op
  private readonly conv1Op:    Op
  private readonly conv2Op:    Op

  constructor(backend: Backend, deep: Tensor, skip: Tensor, w: DecoderBlockWeights, params: DecoderBlockParams) {
    this.inputs = [deep, skip]

    // Fused bilinear upsample + channel concat
    this.upConcatOp = backend.ops.UpsampleConcat(deep, skip, { outH: skip.h, outW: skip.w })

    this.conv1Op = backend.ops.Conv2d(this.upConcatOp.output, w.conv1, {
      outChannels: params.outChannels,
      kernel:      3,
      stride:      1,
      padding:     1,
      activation:  'relu6',
    })

    this.conv2Op = backend.ops.Conv2d(this.conv1Op.output, w.conv2, {
      outChannels: params.outChannels,
      kernel:      3,
      stride:      1,
      padding:     1,
      activation:  'relu6',
    })

    this.output = this.conv2Op.output
  }

  run(): void {
    this.upConcatOp.run()
    this.conv1Op.run()
    this.conv2Op.run()
  }
}
