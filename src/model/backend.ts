export interface Tensor {
  readonly h: number
  readonly w: number
  readonly c: number  // always a multiple of 4
}

export interface Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  run(): void
}

export type Activation = 'none' | 'relu6'

export interface Conv2dParams {
  outChannels: number
  kernel: number
  stride: number
  padding: 'same' | 'valid'
  activation: Activation
}

export interface DepthwiseParams {
  kernel: number
  stride: number
  padding: 'same' | 'valid'
  activation: Activation
}

export interface Backend {
  ops: {
    // Core ops
    Conv2d:           (input: Tensor, weights: Tensor, bias: Tensor, params: Conv2dParams) => Op
    DepthwiseConv2d:  (input: Tensor, weights: Tensor, bias: Tensor, params: DepthwiseParams) => Op
    Add:              (inputs: [Tensor, Tensor]) => Op
    Upsample:         (input: Tensor, outH: number, outW: number) => Op
    Concat:           (inputs: [Tensor, Tensor]) => Op
    Sigmoid:          (input: Tensor) => Op
    // Fused ops
    Conv2dAdd:        (input: Tensor, residual: Tensor, weights: Tensor, bias: Tensor, params: Conv2dParams) => Op
    UpsampleConcat:   (input: Tensor, skip: Tensor, outH: number, outW: number) => Op
    UpsampleSigmoid:  (input: Tensor, outH: number, outW: number) => Op
    UpsampleConv1x1:  (input: Tensor, weights: Tensor, bias: Tensor, activation: Activation, outH: number, outW: number, outChannels: number) => Op
  }

  // Upload a float32 array to GPU (used for weights at model init).
  upload(data: Float32Array): Tensor

  destroy(): void
}
