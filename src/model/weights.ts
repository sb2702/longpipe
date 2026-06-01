export interface Conv2DWeights {
  weights: ArrayLike<number>
  bias:    ArrayLike<number>
}

export interface DepthwiseWeights {
  weights: ArrayLike<number>
  bias:    ArrayLike<number>
}

export interface DepthwiseSeparableWeights {
  dw: DepthwiseWeights
  pw: Conv2DWeights
}

export interface MBConvWeights {
  expand: Conv2DWeights
  dw:     DepthwiseWeights
  proj:   Conv2DWeights
}

export interface DecoderBlockWeights {
  conv1: Conv2DWeights
  conv2: Conv2DWeights
}

export interface UNetWrapperWeights {
  // Down-path: down1 always present; down2 only for two-stage variants (E/C/D).
  down1:        Conv2DWeights
  down2?:       Conv2DWeights
  adapter:      Conv2DWeights
  // Up-path: expand_feat always present; up1Combine only for two-stage variants.
  expandFeat:   Conv2DWeights
  up1Combine?:  Conv2DWeights
  // Final combine + 1-channel output (named `upCombine` for one-stage, `up2Combine`
  // for two-stage — same role, kept as one field).
  upCombine:    Conv2DWeights
}

export interface ConvGRUWeights {
  // Gates conv (2c → 2c) is pre-split at export time into z_conv and r_conv
  // (each 2c → c) — numerically identical to PyTorch's gates(...).chunk(2)
  // along the output-channel dim. Avoids needing a custom split+activate op.
  zConv: Conv2DWeights
  rConv: Conv2DWeights
  // Candidate conv (2c → c).
  cand:  Conv2DWeights
  // Per-recurrent-channel residual scale γ, length = recurrent_ch.
  gamma: ArrayLike<number>
}

export interface ModelWeights {
  encoder: {
    stem: Conv2DWeights
    s0:   DepthwiseSeparableWeights
    s1:   MBConvWeights[]
    s2:   MBConvWeights[]
    s3:   MBConvWeights[]
    s4:   MBConvWeights[]
    s5?:  MBConvWeights[]
    s6?:  MBConvWeights[]
  }
  bottleneck: Conv2DWeights
  decoder: {
    blocks:        DecoderBlockWeights[]
    finalUpsample: Conv2DWeights
    outputConv:    Conv2DWeights
  }
}
