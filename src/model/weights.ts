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
