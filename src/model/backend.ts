export interface Tensor {
  readonly h: number;
  readonly w: number;
  readonly c: number; // always a multiple of 4
}

// Flat parameter buffer — weights, bias, etc. No spatial dimensions.
export interface MLBuffer {}

export interface Op {
  readonly inputs: Tensor[];
  readonly output: Tensor;
  run(): void;
}

export type Activation = "none" | "relu6";

export interface Conv2dParams {
  outChannels: number;
  kernel: number;
  stride: number;
  padding: number | "same" | "valid";
  activation: Activation;
}

export interface DepthwiseParams {
  kernel: number;
  stride: number;
  padding: number | "same" | "valid";
  activation: Activation;
}

export interface UpsampleParams {
  outH: number;
  outW: number;
}

export interface UpsampleConv1x1Params {
  outH:        number;
  outW:        number;
  outChannels: number;
  activation:  Activation;
}

export interface Backend {
  // Allocate a spatial activation buffer. Pass data to pre-fill (tests / first layer).
  tensor(h: number, w: number, c: number, data?: Float32Array): Tensor;

  // Upload a flat parameter buffer (weights, bias).
  upload(data: Float32Array): MLBuffer;

  ops: {
    // Core
    Conv2d:          (input: Tensor, weights: MLBuffer, bias: MLBuffer, params: Conv2dParams) => Op;
    DepthwiseConv2d: (input: Tensor, weights: MLBuffer, bias: MLBuffer, params: DepthwiseParams) => Op;
    Add:             (a: Tensor, b: Tensor) => Op;
    Sigmoid:         (input: Tensor) => Op;
    BilinearUpsample:(input: Tensor, params: UpsampleParams) => Op;
    ChannelConcat:   (a: Tensor, b: Tensor) => Op;

    // Fused — eliminate intermediate buffers between paired ops
    Conv2dAdd:        (input: Tensor, skip: Tensor, weights: MLBuffer, bias: MLBuffer, params: Conv2dParams) => Op;
    UpsampleConcat:   (a: Tensor, b: Tensor, params: UpsampleParams) => Op;
    UpsampleConv1x1:  (input: Tensor, weights: MLBuffer, bias: MLBuffer, params: UpsampleConv1x1Params) => Op;
    UpsampleSigmoid:  (input: Tensor, params: UpsampleParams) => Op;
  };

  destroy(): void;
}
