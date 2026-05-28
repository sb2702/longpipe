import type { Conv2DWeights, DepthwiseWeights } from '~/model/weights.ts'

export type Dtype = 'f32' | 'f16'

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

// External image source for the Input op. ImageBitmap is the static / test
// path (one-shot copy); VideoFrame is the production path (zero-copy on
// WebGPU via importExternalTexture). Both work directly with WebGL2's
// texImage2D.
export type ImageSource = ImageBitmap | VideoFrame;

// Input op produces a Tensor at a fixed (h, w, 4) target resolution. Source
// is set per-frame with setSource(); the output tensor is stable across
// frames (its contents are overwritten in place). Caller pattern:
//   inputOp.setSource(frame); inputOp.run();
// then downstream ops read inputOp.output.
export interface InputOp {
  readonly output: Tensor;
  setSource(src: ImageSource): void;
  run(): void;
}

// Initial data for tensor() and parameters for upload() may arrive as Float32
// (fp32 source) or Uint16 (raw fp16 bits, from a loaded .f16.bin). Backends
// convert as needed to match their own dtype.
export type DataView_ = Float32Array | Uint16Array;

export interface Backend {
  // Numeric precision for activation / weight storage and (on WebGPU) compute.
  readonly dtype: Dtype;

  // The canvas the backend renders to. RenderOp reads its dimensions to size
  // the display Input op + compositor output. Both backends require a canvas
  // at create() time per project_backend_canvas_contract.md.
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;

  // Allocate a spatial activation buffer. Pass data to pre-fill (tests / first layer).
  tensor(h: number, w: number, c: number, data?: DataView_): Tensor;

  // Upload a flat parameter buffer — used internally by ops.
  upload(data: DataView_): MLBuffer;

  ops: {
    // Core
    Conv2d:          (input: Tensor, weights: Conv2DWeights,    params: Conv2dParams)    => Op;
    DepthwiseConv2d: (input: Tensor, weights: DepthwiseWeights, params: DepthwiseParams) => Op;
    Add:             (a: Tensor, b: Tensor) => Op;
    Sigmoid:         (input: Tensor) => Op;
    BilinearUpsample:(input: Tensor, params: UpsampleParams) => Op;
    BicubicUpsample: (input: Tensor, params: UpsampleParams) => Op;
    ChannelConcat:   (a: Tensor, b: Tensor) => Op;

    // ConvGRU + wrapper primitives (temporal models)
    Tanh:            (input: Tensor) => Op;
    ElementwiseMul:  (a: Tensor, b: Tensor) => Op;
    GruUpdate:       (z: Tensor, h_prev: Tensor, h_til: Tensor) => Op;
    GammaResidual:   (b: Tensor, h_new: Tensor, gamma: ArrayLike<number>) => Op;

    // Fused — eliminate intermediate buffers between paired ops
    Conv2dAdd:       (input: Tensor, skip: Tensor, weights: Conv2DWeights,    params: Conv2dParams)          => Op;
    UpsampleConcat:  (a: Tensor, b: Tensor, params: UpsampleParams) => Op;
    UpsampleConv1x1: (input: Tensor, weights: Conv2DWeights,                  params: UpsampleConv1x1Params) => Op;
    UpsampleSigmoid: (input: Tensor, params: UpsampleParams) => Op;

    // Image source ingestion. Bilinear-resamples the source down to (h, w, 4).
    Input:           (h: number, w: number) => InputOp;
  };

  // Render-to-display ops. Produce no Tensor — write directly to the canvas.
  presenters: {
    CompositeSolid:          (image: Tensor, alpha: Tensor, bgColor: [number, number, number]) => Presenter;
    CompositeImage:          (image: Tensor, alpha: Tensor, bg: Tensor) => Presenter;
    // Same as CompositeImage but bg may be smaller than (image, alpha) — bg
    // is bilinearly sampled. Used by CompositorBlur to absorb the final
    // pyramid upsample into this pass for free.
    CompositeImageBilinear:  (image: Tensor, alpha: Tensor, bg: Tensor) => Presenter;
    // Passthrough: writes image directly to canvas; no alpha, no bg. Used
    // by RenderOp when the renderer is disabled (true GPU-level passthrough
    // — input frame in, same frame on the canvas).
    CompositePassthrough:    (image: Tensor) => Presenter;
  };

  // Read tensor data back to host as fp32. The tensor must have been allocated
  // by this backend; conversion from fp16 storage is handled internally.
  readback(tensor: Tensor): Promise<Float32Array>;

  // Wait for all pending GPU work to complete. Cheaper than readback when
  // you only need a sync barrier (e.g., timing benchmarks). WebGPU uses
  // queue.onSubmittedWorkDone(); WebGL2 uses fenceSync + clientWaitSync
  // or gl.finish() as fallback.
  sync(): Promise<void>;

  destroy(): void;
}
