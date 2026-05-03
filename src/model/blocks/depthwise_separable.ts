import type { Backend, Tensor, Op } from '~/model/backend'
import type { DepthwiseSeparableWeights } from '~/model/weights'

export interface DepthwiseSeparableParams {
  outChannels: number
  kernel:      number
  stride:      number
  padding:     number | 'same' | 'valid'
}

export class DepthwiseSeparable implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  private readonly dwOp: Op
  private readonly pwOp: Op

  constructor(backend: Backend, input: Tensor, w: DepthwiseSeparableWeights, params: DepthwiseSeparableParams) {
    this.inputs = [input]

    this.dwOp = backend.ops.DepthwiseConv2d(input, w.dw, {
      kernel:     params.kernel,
      stride:     params.stride,
      padding:    params.padding,
      activation: 'relu6',
    })

    this.pwOp = backend.ops.Conv2d(this.dwOp.output, w.pw, {
      outChannels: params.outChannels,
      kernel:      1,
      stride:      1,
      padding:     0,
      activation:  'none',
    })

    this.output = this.pwOp.output
  }

  run(): void {
    this.dwOp.run()
    this.pwOp.run()
  }
}
