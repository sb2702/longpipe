import type { Backend, Tensor, MLBuffer, Op } from '~/model/backend'

export interface MBConvWeights {
  expandWeights?: MLBuffer  // omit when expandRatio === 1
  expandBias?:    MLBuffer
  dwWeights:      MLBuffer
  dwBias:         MLBuffer
  projWeights:    MLBuffer
  projBias:       MLBuffer
}

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
  readonly addOp:    Op | null

  constructor(backend: Backend, input: Tensor, w: MBConvWeights, params: MBConvParams) {
    this.inputs = [input]

    const hasExpand   = params.midChannels !== params.inChannels
    const hasResidual = params.stride === 1 && params.inChannels === params.outChannels

    // 1. Expand (1x1, relu6) — null when expandRatio === 1
    this.expandOp = hasExpand
      ? backend.ops.Conv2d(input, w.expandWeights!, w.expandBias!, {
          outChannels: params.midChannels,
          kernel:      1,
          stride:      1,
          padding:     0,
          activation:  'relu6',
        })
      : null

    const expanded = this.expandOp ? this.expandOp.output : input

    // 2. Depthwise (kxk, relu6)
    this.dwOp = backend.ops.DepthwiseConv2d(expanded, w.dwWeights, w.dwBias, {
      kernel:     params.kernel,
      stride:     params.stride,
      padding:    params.padding,
      activation: 'relu6',
    })

    // 3. Project (1x1, no activation)
    this.projOp = backend.ops.Conv2d(this.dwOp.output, w.projWeights, w.projBias, {
      outChannels: params.outChannels,
      kernel:      1,
      stride:      1,
      padding:     0,
      activation:  'none',
    })

    // 4. Residual add — null when stride > 1 or channels change
    this.addOp = hasResidual ? backend.ops.Add(input, this.projOp.output) : null

    this.output = this.addOp ? this.addOp.output : this.projOp.output
  }

  run(): void {
    this.expandOp?.run()
    this.dwOp.run()
    this.projOp.run()
    this.addOp?.run()
  }
}
