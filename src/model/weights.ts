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
