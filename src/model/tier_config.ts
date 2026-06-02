import type { BaseNetworkCtor } from '~/model/tier_model.ts'
import type { UNetWrapperParams } from '~/model/blocks/unet_wrapper.ts'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small.ts'
import { EfficientNetLiteMattingLarge } from '~/model/networks/efficientnetlite_matting_large.ts'
import { EfficientNetLiteMattingXL }    from '~/model/networks/efficientnetlite_matting_xl.ts'

// Production tier table — "the model is code". Per-tier wrapper variant + GRU
// placement live here (NOT in the .bin); the .bin carries only weights. Mirrors
// the variants of the trained checkpoints. All five tiers ship gru-at-base
// (ConvGRU on the c_up carrier at base res) as of 2026-06-02 — xs/small gained
// GRU heads (small/A, xs/B temporal checkpoints) after the static versions read
// poorly in the live demo.
// Production widths are fixed: cHigh = cLow = 4, cUp = 2 (the fused narrow ops
// require them; the carrier lives in a 4-ch tensor's .xy).
export interface Res { w: number; h: number }

export interface TierConfig {
  base:    BaseNetworkCtor
  wrapper: UNetWrapperParams   // variant, cHigh, cLow, cUp, gruAtBase
  hasGru:  boolean             // false = static tier; build TierModel without a gru
  // x_hr (network input) AND alpha-output resolution. The wrapper down-path
  // strides this down to baseRes (canvasRes / canvas_mul, mul set by variant:
  // A=2, B=3, D=2.5, E=4). 16:9 production constants — must match what the
  // checkpoints were exported at (training/deploy/export_sdk_weights.py).
  canvasRes: Res
  // Base-network input resolution. ALSO the hPrev (ConvGRU hidden) resolution
  // for gru-at-base tiers — the carrier the GRU smooths lives at base res.
  baseRes:   Res
}

const SMALL = EfficientNetLiteMattingSmall as unknown as BaseNetworkCtor
const LARGE = EfficientNetLiteMattingLarge as unknown as BaseNetworkCtor
const XL    = EfficientNetLiteMattingXL    as unknown as BaseNetworkCtor

export const TIER_CONFIG: Record<string, TierConfig> = {
  xs:     { base: SMALL, wrapper: { variant: 'B', cHigh: 4, cLow: 4, cUp: 2, gruAtBase: true  }, hasGru: true,  canvasRes: { w: 384,  h: 216 }, baseRes: { w: 128, h: 72  } },
  small:  { base: SMALL, wrapper: { variant: 'A', cHigh: 4, cLow: 4, cUp: 2, gruAtBase: true  }, hasGru: true,  canvasRes: { w: 384,  h: 216 }, baseRes: { w: 192, h: 108 } },
  medium: { base: LARGE, wrapper: { variant: 'A', cHigh: 4, cLow: 4, cUp: 2, gruAtBase: true  }, hasGru: true,  canvasRes: { w: 512,  h: 288 }, baseRes: { w: 256, h: 144 } },
  large:  { base: LARGE, wrapper: { variant: 'D', cHigh: 4, cLow: 4, cUp: 2, gruAtBase: true  }, hasGru: true,  canvasRes: { w: 640,  h: 360 }, baseRes: { w: 256, h: 144 } },
  xl:     { base: XL,    wrapper: { variant: 'E', cHigh: 4, cLow: 4, cUp: 2, gruAtBase: true  }, hasGru: true,  canvasRes: { w: 1280, h: 720 }, baseRes: { w: 320, h: 180 } },
}
