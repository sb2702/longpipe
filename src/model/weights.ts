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

// Native narrow-path packings (each leaf's `.weights` uses the packer named):
//   down1     conv_weights (mat4x4; B feeds DownAdapter, E/D a plain Conv2d)
//   down2     conv_weights (mat4x4; DownAdapter down conv, E/D only)
//   adapter   conv_weights (mat4x4, padded 4→3 → [4,4,1,1])
//   expandFeat conv_expand_weights (mat4x2, feat_ch→2)
//   up1Combine cat_conv_6to2_weights (mat3x2, 6→2; two-stage only)
//   upCombine  up_final_weights (5→1, B/E) | up_final_skip_weights (9→1, D)
export interface UNetWrapperWeights {
  // Down-path: down1 always present; down2 only for two-stage variants (E/D).
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

// Production ConvGRU fused weights (c_up=2, split_ratio=0.5 → passthrough=1,
// recurrent=1), packed for the fused gates_fused / cand_update shaders.
//   gates: gates Conv2d(2,2) → 9 vec4 per kpos = (z_w_b, z_w_h, r_w_b, r_w_h)
//   cand:  cand  Conv2d(2,1) → 9 vec4 per kpos, .xy = (b_w, rh_w)
export interface ConvGRUWeights {
  gates:     ArrayLike<number>   // 9*4 = 36 floats
  gatesBias: ArrayLike<number>   // 2: (z_bias, r_bias)
  cand:      ArrayLike<number>   // 9*4 = 36 floats
  candBias:  ArrayLike<number>   // 1: (cand_bias)
  gamma:     ArrayLike<number>   // 1: recurrent_ch
}

// Optical-flow head weights — the only learned part of the flow net (the matting
// encoder + wrapper are already in ModelWeights / UNetWrapperWeights; the flow net
// rides their cached taps). All convs use the canonical mat4x4 conv layout. Predict
// heads emit 4 ch (flow in .xy, .zw = 0); upflow is the 2→2 (packed 4→4) flow
// upsampler. Order/length mirror training FlowEncoderNet (base/4, no tap-half).
export interface FlowWeights {
  stem:       Conv2DWeights     // 6→decW (packed 8→decW), k7 s2, leaky
  stages:     Conv2DWeights[]   // (decW+tap)→decW, k5/5/3/3 s2, leaky; one per tap
  predictBot: Conv2DWeights     // fused[-1]→4 (flow .xy), k3
  deconv:     Conv2DWeights[]   // ConvTranspose decIn→decW, k4 s2 p1, leaky
  upflow:     Conv2DWeights[]   // ConvTranspose 4→4 (2→2 flow), k4 s2 p1, no act
  predict:    Conv2DWeights[]   // cat→4 (flow .xy), k3; finest = base/4
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
