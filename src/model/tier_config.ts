import type { BaseNetworkCtor } from '~/model/tier_model.ts'
import type { UNetWrapperParams } from '~/model/blocks/unet_wrapper.ts'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small.ts'
import { EfficientNetLiteMattingLarge } from '~/model/networks/efficientnetlite_matting_large.ts'
import { EfficientNetLiteMattingXL }    from '~/model/networks/efficientnetlite_matting_xl.ts'

// Production tier table — "the model is code". Per-tier wrapper variant lives
// here (NOT in the .bin); the .bin carries only weights. Mirrors the variants of
// the trained checkpoints. All five tiers are STATIC — temporal stability comes
// from the optical-flow path (a separate OpticalFlowNet riding these tiers'
// cached encoder taps), not an output ConvGRU.
// Production widths are fixed: cHigh = cLow = 4, cUp = 2 (the fused narrow ops
// require them; the carrier lives in a 4-ch tensor's .xy).
export interface Res { w: number; h: number }

export interface TierConfig {
  base:    BaseNetworkCtor
  wrapper: UNetWrapperParams   // variant, cHigh, cLow, cUp
  // x_hr (network input) AND alpha-output resolution. The wrapper down-path
  // strides this down to baseRes (canvasRes / canvas_mul, mul set by variant:
  // A=2, B=3, D=2.5, E=4). Production constants mirroring train_run.py PRESETS
  // 'landscape' shapes (no longer exactly 16:9 — e.g. large is 256×160 = 8:5) —
  // must match what the checkpoints were exported at (export_sdk_weights.py).
  canvasRes: Res
  // Base-network input resolution (the wrapper's adapted base input). Also the
  // resolution the optical-flow net predicts at (base/4) before warp-res upsample.
  baseRes:   Res
  // Optical-flow head fuses the /2 tap at the stem (predicts base/2). XS only.
  // The flow net is wired iff the loaded .bin carries a `flow` blob.
  flowFuseStem?: boolean
}

const SMALL = EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor
const LARGE = EfficientNetLiteMattingLarge as unknown as BaseNetworkCtor
const XL    = EfficientNetLiteMattingXL    as unknown as BaseNetworkCtor

export const TIER_CONFIG: Record<string, TierConfig> = {
  xs:     { base: SMALL, wrapper: { variant: 'B', cHigh: 4, cLow: 4, cUp: 2 }, canvasRes: { w: 384,  h: 240 }, baseRes: { w: 128, h: 80  }, flowFuseStem: true },
  small:  { base: SMALL, wrapper: { variant: 'A', cHigh: 4, cLow: 4, cUp: 2 }, canvasRes: { w: 384,  h: 224 }, baseRes: { w: 192, h: 112 } },
  medium: { base: LARGE, wrapper: { variant: 'A', cHigh: 4, cLow: 4, cUp: 2 }, canvasRes: { w: 512,  h: 320 }, baseRes: { w: 256, h: 160 } },
  large:  { base: LARGE, wrapper: { variant: 'D', cHigh: 4, cLow: 4, cUp: 2 }, canvasRes: { w: 640,  h: 400 }, baseRes: { w: 256, h: 160 } },
  xl:     { base: XL,    wrapper: { variant: 'E', cHigh: 4, cLow: 4, cUp: 2 }, canvasRes: { w: 1280, h: 768 }, baseRes: { w: 320, h: 192 } },
}
