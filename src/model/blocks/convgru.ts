import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { ConvGRUWeights } from '~/model/weights.ts'

// Production ConvGRU (c_up=2, split_ratio=0.5 → passthrough=1, recurrent=1),
// built from the two fused dispatches GatesFused + CandUpdateFused.
//
// Channel carrier layout (4-channel tensor, only leading lanes used):
//   input  uIn   : .x = passthrough a, .y = recurrent b
//   state  hPrev : .z = previous hidden h (zero tensor on frame 0)
//   output       : .x = a, .y = b_out, .z = h_new
//
// The output tensor doubles as next frame's hPrev (hidden in .z) — the runtime
// ping-pongs two output buffers across frames, no separate hidden-state buffer.
//
// Inputs:
//   uIn   : c_up=2 feature at canvas resolution (.x=a, .y=b)
//   hPrev : hidden carrier from the previous frame (.z); zero buffer on frame 0
//
// Output:
//   .output : (a, b_out, h_new, 0) — .xy feeds downstream, .z threads forward
export class ConvGRU implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  private readonly gatesOp: Op
  private readonly candOp:  Op

  constructor(backend: Backend, uIn: Tensor, hPrev: Tensor, w: ConvGRUWeights) {
    this.inputs = [uIn, hPrev]

    this.gatesOp = backend.ops.GatesFused(uIn, hPrev, {
      weights: w.gates,
      bias:    w.gatesBias,
    })

    this.candOp = backend.ops.CandUpdateFused(uIn, hPrev, this.gatesOp.output, {
      weights: w.cand,
      bias:    w.candBias,
    }, w.gamma)

    this.output = this.candOp.output
  }

  run(): void {
    this.gatesOp.run()
    this.candOp.run()
  }
}
