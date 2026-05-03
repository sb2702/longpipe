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

export interface Backend {
  // Allocate a spatial activation buffer. Pass data to pre-fill (tests / first layer).
  tensor(h: number, w: number, c: number, data?: Float32Array): Tensor;

  // Upload a flat parameter buffer (weights, bias).
  upload(data: Float32Array): MLBuffer;

  ops: {
    Conv2d: (input: Tensor, weights: MLBuffer, bias: MLBuffer, params: Conv2dParams) => Op;
  };

  destroy(): void;
}
