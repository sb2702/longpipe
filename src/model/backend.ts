import type { Conv2DWeights, DepthwiseWeights } from '~/model/weights'

export interface Tensor {
  readonly h: number;
  readonly w: number;
  readonly c: number; // always a multiple of 4
}

// Flat parameter buffer — internal to backends, not part of the public API.
export interface MLBuffer {}

export interface Op {
  readonly inputs: Tensor[];
  readonly output: Tensor;
  run(): void;
}

// Renders to the backend's canvas (no Tensor output). Each backend's factory
// hides per-frame setup (e.g. WebGPU swapchain texture acquisition).
export interface Presenter {
  run(): void;
}

export type Activation = "none" | "relu6";

export interface Conv2dParams {
  outChannels: number;
  kernel:      number;
  stride:      number;
  padding:     number | "same" | "valid";
  activation:  Activation;
}

export interface DepthwiseParams {
  kernel:     number;
  stride:     number;
  padding:    number | "same" | "valid";
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

export interface GaussianBlur1DParams {
  direction: "horizontal" | "vertical";
  sigma:     number;
}

export interface Backend {
  // Allocate a spatial activation buffer. Pass data to pre-fill (tests / first layer).
  tensor(h: number, w: number, c: number, data?: Float32Array): Tensor;

  // Upload a flat parameter buffer — used internally by ops.
  upload(data: Float32Array): MLBuffer;

  ops: {
    // Core
    Conv2d:          (input: Tensor, weights: Conv2DWeights,    params: Conv2dParams)    => Op;
    DepthwiseConv2d: (input: Tensor, weights: DepthwiseWeights, params: DepthwiseParams) => Op;
    Add:             (a: Tensor, b: Tensor) => Op;
    Sigmoid:         (input: Tensor) => Op;
    BilinearUpsample:(input: Tensor, params: UpsampleParams) => Op;
    ChannelConcat:   (a: Tensor, b: Tensor) => Op;

    // Fused — eliminate intermediate buffers between paired ops
    Conv2dAdd:       (input: Tensor, skip: Tensor, weights: Conv2DWeights,    params: Conv2dParams)          => Op;
    UpsampleConcat:  (a: Tensor, b: Tensor, params: UpsampleParams) => Op;
    UpsampleConv1x1: (input: Tensor, weights: Conv2DWeights,                  params: UpsampleConv1x1Params) => Op;
    UpsampleSigmoid: (input: Tensor, params: UpsampleParams) => Op;

    // Effects
    GaussianBlur1D:  (input: Tensor, params: GaussianBlur1DParams) => Op;
  };

  // Render-to-display ops. Produce no Tensor — write directly to the canvas.
  presenters: {
    CompositeSolid: (image: Tensor, alpha: Tensor, bgColor: [number, number, number]) => Presenter;
    CompositeImage: (image: Tensor, alpha: Tensor, bg: Tensor) => Presenter;
  };

  // Read tensor data back to host. The tensor must have been allocated by this backend.
  readback(tensor: Tensor): Promise<Float32Array>;

  destroy(): void;
}
