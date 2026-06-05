import type { Backend, Tensor, Op } from '~/model/backend.ts'
import type { UNetWrapperWeights } from '~/model/weights.ts'

export type UNetVariant = 'A' | 'B' | 'E' | 'D'

export interface UNetWrapperParams {
  variant: UNetVariant
  // Channel widths in the wrapper's stages. Match training/models/unet_model.py.
  // The fused narrow ops require the production widths: c_high = c_low = 4,
  // c_up = 2 (the carrier lives in a 4-ch tensor's .xy).
  cHigh:   number   // down1 output (E/D); down conv output (A/B). Expected 4.
  cLow:    number   // adapter input. Expected 4.
  cUp:     number   // expand_feat / up1_combine output. Expected 2.
}

// Result of the wrapper down-path. `adapted` is the base input (vec4 .xyz);
// d1 / dFull are skips the up-path consumes; mid{H,W} is the two-stage skip res.
export interface WrapperDown {
  steps:   Op[]
  adapted: Tensor
  d1:      Tensor | null
  dFull:   Tensor | null
  midH:    number
  midW:    number
}

// Builds the wrapper down-path (x_hr → adapted base input). mid res is computed
// from canvas (NOT feat), so this runs before any feature exists — letting the
// full base+wrapper composition slot the base between down and up.
export function buildWrapperDown(
  backend: Backend, x_hr: Tensor, w: UNetWrapperWeights, params: UNetWrapperParams,
): WrapperDown {
  const variant = params.variant
  const isTwoStage = variant === 'E' || variant === 'D'
  const hasSkip = variant === 'D'
  const canvasH = x_hr.h, canvasW = x_hr.w
  const steps: Op[] = []
  let d1: Tensor | null = null
  let dFull: Tensor | null = null
  let midH = 0, midW = 0
  let adapted: Tensor

  if (!isTwoStage) {
    // A/B: down1 + adapter fused (single strided conv → adapter). A=stride 2,
    // B=stride 3 (one-stage down_ratios [2.0] / [3.0]).
    const stride = variant === 'A' ? 2 : 3
    const da = backend.ops.DownAdapter(x_hr, w.down1, w.adapter, { stride })
    steps.push(da)
    adapted = da.output
  } else if (hasSkip) {
    // D: down1 stride-1 at full canvas res → d_full; bilinear resize → d1
    //    (mid = round(canvas/1.25)); down2 + adapter fused (stride 2).
    if (!w.down2) throw new Error('variant D requires w.down2')
    midH = Math.round(canvasH / 1.25)
    midW = Math.round(canvasW / 1.25)
    const down1 = backend.ops.Conv2d(x_hr, w.down1, {
      outChannels: params.cHigh, kernel: 3, stride: 1, padding: 1, activation: 'relu',
    })
    steps.push(down1)
    dFull = down1.output
    const resize = backend.ops.BilinearUpsample(dFull, { outH: midH, outW: midW })
    steps.push(resize)
    d1 = resize.output
    const da = backend.ops.DownAdapter(d1, w.down2, w.adapter, { stride: 2 })
    steps.push(da)
    adapted = da.output
  } else {
    // E: down1 stride-2 → d1 (mid = down1 output res); down2 + adapter (stride 2).
    if (!w.down2) throw new Error('variant E requires w.down2')
    const down1 = backend.ops.Conv2d(x_hr, w.down1, {
      outChannels: params.cHigh, kernel: 3, stride: 2, padding: 1, activation: 'relu',
    })
    steps.push(down1)
    d1 = down1.output
    midH = d1.h; midW = d1.w
    const da = backend.ops.DownAdapter(d1, w.down2, w.adapter, { stride: 2 })
    steps.push(da)
    adapted = da.output
  }

  return { steps, adapted, d1, dFull, midH, midW }
}

// Builds the wrapper up-path (feat + skips → sigmoid alpha). `feat` is the base's
// pre-head feature at base-input resolution (forward_features output).
export function buildWrapperUp(
  backend: Backend, x_hr: Tensor, feat: Tensor,
  d1: Tensor | null, dFull: Tensor | null, midH: number, midW: number,
  w: UNetWrapperWeights, params: UNetWrapperParams,
): { steps: Op[]; alpha: Tensor } {
  const variant = params.variant
  const isTwoStage = variant === 'E' || variant === 'D'
  const hasSkip = variant === 'D'
  const steps: Op[] = []

  const expand = backend.ops.ConvExpand(feat, w.expandFeat)   // → c_up=2 carrier at base
  steps.push(expand)
  const carrier = expand.output                               // c_up=2 at base resolution

  let uAtCanvas: Tensor
  if (isTwoStage) {
    if (!w.up1Combine) throw new Error('two-stage variant requires w.up1Combine')
    const u_a_up = backend.ops.BilinearUpsample(carrier, { outH: midH, outW: midW })
    steps.push(u_a_up)
    const u1 = backend.ops.CatConv6to2(u_a_up.output, d1!, w.up1Combine)   // concat + 6→2
    steps.push(u1)
    const u1_up = backend.ops.BilinearUpsample(u1.output, { outH: x_hr.h, outW: x_hr.w })
    steps.push(u1_up)
    uAtCanvas = u1_up.output
  } else {
    const u_a_up = backend.ops.BilinearUpsample(carrier, { outH: x_hr.h, outW: x_hr.w })
    steps.push(u_a_up)
    uAtCanvas = u_a_up.output
  }

  const head = hasSkip
    ? backend.ops.UpFinalSkip(uAtCanvas, dFull!, x_hr, w.upCombine)   // 9→1 (D)
    : backend.ops.UpFinal(uAtCanvas, x_hr, w.upCombine)              // 5→1 (A/B/E)
  steps.push(head)

  return { steps, alpha: head.output }
}

// Static-only UNet wrapper on the native narrow path (c_up=2 carrier in .xy).
// The base is a black box — feat_lr provided externally. The output GRU is
// identity-at-init in the static fixtures, so it is omitted here.
//
// Variants: A/B one-stage (DownAdapter); E/D two-stage (CatConv6to2 up1); D adds
// the full-res d_full skip + a 9→1 (UpFinalSkip) head. See buildWrapperDown/Up.
//
// Inputs:  x_hr (RGB padded to 4 ch at canvas res), feat_lr (base pre-head feat).
// Output:  .output — 4-ch tensor with sigmoid alpha in channel 0.
export class UNetWrapper implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor

  private readonly steps: Op[]

  constructor(
    backend: Backend,
    x_hr: Tensor,
    feat_lr: Tensor,
    w: UNetWrapperWeights,
    params: UNetWrapperParams,
  ) {
    this.inputs = [x_hr, feat_lr]
    const down = buildWrapperDown(backend, x_hr, w, params)
    const up   = buildWrapperUp(backend, x_hr, feat_lr, down.d1, down.dFull, down.midH, down.midW, w, params)
    this.steps = [...down.steps, ...up.steps]
    this.output = up.alpha
  }

  run(): void {
    for (const op of this.steps) op.run()
  }
}
