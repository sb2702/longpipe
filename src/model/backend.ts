export interface Tensor {
  readonly h: number;
  readonly w: number;
  readonly c: number; // always a multiple of 4
}

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
  ops: {
    // Core ops
    Conv2d: (
      input: Tensor,
      weights: Tensor,
      bias: Tensor,
      params: Conv2dParams,
    ) => Op;
  };

  // Upload a float32 array to GPU (used for weights at model init).
  upload(data: Float32Array): Tensor;

  destroy(): void;
}
