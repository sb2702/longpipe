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

export type Activation = "none" | "relu6" | "relu";

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

export interface ProjResidualParams {
  outChannels: number;
}

export interface ConcatConv2dParams {
  outChannels: number;
}

export interface DownAdapterParams {
  stride: number;
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

    // Generic elementwise primitives (temporal models / effects)
    Tanh:            (input: Tensor) => Op;
    ElementwiseMul:  (a: Tensor, b: Tensor) => Op;

    // Fused ConvGRU (production config c_up=2, recurrent=1). GatesFused emits
    // (z, r); CandUpdateFused consumes it + does candidate/update/output.
    GatesFused:      (uIn: Tensor, hPrev: Tensor, weights: Conv2DWeights) => Op;
    CandUpdateFused: (uIn: Tensor, hPrev: Tensor, gatesOut: Tensor, weights: Conv2DWeights, gamma: ArrayLike<number>) => Op;

    // Bespoke N→2 conv 3×3 + relu (wrapper expand_feat). Output is the c_up=2
    // carrier (.xy = 2 native channels, .zw = 0).
    ConvExpand:      (input: Tensor, weights: Conv2DWeights) => Op;
    // Fused concat(u, d) + 6→2 conv 3×3 + relu (E up1_combine). u = c_up=2
    // carrier (.xy), d = c_high=4 skip (full vec4). Output c_up=2 carrier.
    CatConv6to2:     (u: Tensor, d: Tensor, weights: Conv2DWeights) => Op;
    // Fused stride-N 3×3 conv (4→4) + relu + 1×1 adapter (4→3) → base input
    // (wrapper down2+adapter, or down1+adapter for A/B). Output vec4(xyz, 0).
    DownAdapter:     (input: Tensor, downWeights: Conv2DWeights, adaptWeights: Conv2DWeights, params: DownAdapterParams) => Op;
    // Alpha heads: fused concat → conv 3×3 → sigmoid. UpFinal (A/B, 5→1) takes
    // [u, rgb]; UpFinalSkip (C/D, 9→1) takes [u, d_full, rgb]. Output .x = alpha.
    UpFinal:         (u: Tensor, rgb: Tensor, weights: Conv2DWeights) => Op;
    UpFinalSkip:     (u: Tensor, dFull: Tensor, rgb: Tensor, weights: Conv2DWeights) => Op;

    // Fused — eliminate intermediate buffers between paired ops
    Conv2dAdd:       (input: Tensor, skip: Tensor, weights: Conv2DWeights,    params: Conv2dParams)          => Op;
    // Bespoke 1×1 proj + residual add (MBConv tail). Specializes Conv2dAdd to
    // kernel=1/stride=1/pad=0/no-activation — drops the kernel loop entirely.
    ProjResidual:    (input: Tensor, skip: Tensor, weights: Conv2DWeights,    params: ProjResidualParams)    => Op;
    // Bespoke concat(a,b) → 3×3 conv (pad 1) → relu6, fused (decoder conv1).
    // Inputs must share resolution; conv weight in-channels ordered [a, b].
    ConcatConv2d:    (a: Tensor, b: Tensor, weights: Conv2DWeights,           params: ConcatConv2dParams)    => Op;
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
