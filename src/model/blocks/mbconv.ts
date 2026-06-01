import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { MBConvWeights } from '~/model/weights.ts'

export interface MBConvParams {
  inChannels:  number
  midChannels: number  // inChannels * expandRatio
  outChannels: number
  kernel:      number
  stride:      number
  padding:     number | 'same' | 'valid'
}

export class MBConv implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  readonly expandOp: Op | null
  readonly dwOp:     Op
  readonly projOp:   Op

  constructor(backend: Backend, input: Tensor, w: MBConvWeights, params: MBConvParams) {
    this.inputs = [input]

    const hasExpand   = params.midChannels !== params.inChannels
    const hasResidual = params.stride === 1 && params.inChannels === params.outChannels

    // 1. Expand (1x1, relu6) — null when expandRatio === 1
    this.expandOp = hasExpand
      ? backend.ops.Conv2d(input, w.expand, {
          outChannels: params.midChannels,
          kernel:      1,
          stride:      1,
          padding:     0,
          activation:  'relu6',
        })
      : null

    const expanded = this.expandOp ? this.expandOp.output : input

    // 2. Depthwise (kxk, relu6)
    this.dwOp = backend.ops.DepthwiseConv2d(expanded, w.dw, {
      kernel:     params.kernel,
      stride:     params.stride,
      padding:    params.padding,
      activation: 'relu6',
    })

    // 3. Project (1x1, no activation). For residual blocks the skip add is
    //    fused into the projection (ProjResidual — bespoke 1×1 + residual);
    //    otherwise a plain 1×1 Conv2d.
    this.projOp = hasResidual
      ? backend.ops.ProjResidual(this.dwOp.output, input, w.proj, {
          outChannels: params.outChannels,
        })
      : backend.ops.Conv2d(this.dwOp.output, w.proj, {
          outChannels: params.outChannels,
          kernel:      1,
          stride:      1,
          padding:     0,
          activation:  'none',
        })

    this.output = this.projOp.output
  }

  run(): void {
    this.expandOp?.run()
    this.dwOp.run()
    this.projOp.run()
  }
}
