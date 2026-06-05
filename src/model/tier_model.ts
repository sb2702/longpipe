import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { ModelWeights } from '~/model/weights.ts'
import type { UNetWrapperWeights } from '~/model/weights.ts'
import {
  buildWrapperDown, buildWrapperUp, type UNetWrapperParams,
} from '~/model/blocks/unet_wrapper.ts'

// A base network: input → encoder/decoder → exposes `featLowRes` (pre-head
// feature). The 5 production tiers reuse 3 base classes (small/large/xl).
export interface BaseNetwork {
  readonly featLowRes: Tensor
  // Encoder pyramid (/4../32, finest→coarsest), computed on the adapted input —
  // exposed so the optical-flow net can ride the cached activations next frame.
  readonly encoderTaps: Tensor[]
  // /2 tap — present only on small-encoder networks (XS tap-half flow head).
  readonly halfTap?: Tensor
  run(): void
}
export type BaseNetworkCtor = new (backend: Backend, input: Tensor, w: ModelWeights) => BaseNetwork

// Full production tier model: wrapper down-path → base → up-path. Mirrors
// training UNetMattingModel.forward:
//   x_hr → down_path → adapted → base.forward_features → feat (at base-input
//   res, via interpolate) → up_path → sigmoid alpha at canvas res.
//
// Output GRU is omitted (static; identity-at-init in fixtures) — matches the
// UNetWrapper static path. `.output` is a 4-ch tensor with alpha in channel 0.
export class TierModel implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  // Base encoder pyramid (finest→coarsest), computed on the adapted frame — the
  // optical-flow net reads these to ride the cached matting activations.
  readonly encoderTaps: Tensor[]
  // /2 tap (small-encoder tiers only) — the XS tap-half flow head fuses it.
  readonly halfTap?: Tensor

  private readonly steps: { run(): void }[]

  constructor(
    backend: Backend,
    x_hr: Tensor,
    baseWeights: ModelWeights,
    wrapperWeights: UNetWrapperWeights,
    params: UNetWrapperParams,
    BaseCtor: BaseNetworkCtor,
  ) {
    this.inputs = [x_hr]

    const down = buildWrapperDown(backend, x_hr, wrapperWeights, params)

    // Base consumes the adapter output; produces a pre-head feature.
    const base = new BaseCtor(backend, down.adapted, baseWeights)

    // forward_features interpolates the feature to base-input res before the
    // wrapper consumes it (mirrors training/models forward_features).
    const featUp = backend.ops.BilinearUpsample(base.featLowRes, {
      outH: down.adapted.h,
      outW: down.adapted.w,
    })

    const up = buildWrapperUp(
      backend, x_hr, featUp.output, down.d1, down.dFull, down.midH, down.midW,
      wrapperWeights, params,
    )

    this.steps = [...down.steps, base, featUp, ...up.steps]
    this.output = up.alpha
    this.encoderTaps = base.encoderTaps
    this.halfTap = base.halfTap
  }

  run(): void {
    for (const s of this.steps) s.run()
  }
}
