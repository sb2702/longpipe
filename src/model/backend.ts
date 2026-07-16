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

// Which attached canvas a presenter writes to. 'main' is the always-present
// output canvas (configured at create()); 'preview' is an optional second
// canvas registered via attachCanvas() for the preview-effect feature.
// WebGPU routes each target to its own GPUCanvasContext; WebGL ignores the
// target (single context — preview is presented by the renderer via a
// snapshot of the main canvas). Defaults to 'main' everywhere so existing
// single-output callers are unaffected.
export type RenderTarget = "main" | "preview";

export type Activation = "none" | "relu6" | "relu" | "leaky";

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

export interface ConvTranspose2dParams {
  outChannels: number;
  kernel:      number;
  stride:      number;
  padding:     number;   // symmetric; output_padding is 0
  activation:  Activation;
}

export interface WarpParams {
  // flow.xy is multiplied by this before sampling. Folds the backward-warp
  // negation (gather from a forward flow) and any base→warp magnitude rescale
  // into one constant (e.g. -(warpW / baseW)).
  flowScale: number;
}

export interface StabilizeParams {
  tLo:     number;   // gate opens above this flow magnitude
  tHi:     number;   // gate fully open (trust fresh pred) at/above this
  leak:    number;   // floor on the pred weight even where static (lets it heal)
  release: number;   // envelope decay per frame (peak-hold attack/slow release)
  // Occlusion-seam gate: the gate ALSO opens where the flow field diverges (an
  // occlusion/disocclusion boundary the magnitude gate is blind to, since the
  // revealed background is static). |div(flow)| is a finite-difference over a step
  // that spans ~1 base/4 pixel (stepX/stepY ≈ canvas/flow resolution ratio).
  tDiv:     number;  // divergence above which the seam gate starts opening
  divScale: number;  // soft-gate width for the divergence term
  stepX:    number;  // finite-difference step (px) in x / y
  stepY:    number;
}

export interface UpsampleParams {
  outH: number;
  outW: number;
}

// Face-box decode from the 5-keypoint heatmaps (FaceHeatmapNet output). Peaks
// are refined with a windowed soft-argmax centroid (±win cells) — NEVER hard
// argmax (whole-cell snapping jitters the crop; see project_face_keypoint_head).
export interface FaceBoxParams {
  win:      number;   // soft-argmax window, cells around the peak (training PoC: 3)
  thresh:   number;   // min per-keypoint peak score; below → box score = 0
  boxScale: number;   // crop side = boxScale × keypoint-hull long side (px-square)
}

// MULTI-FACE decode → a 1×K×4 box tensor (K = maxFaces), one (cx, cy, halfSide,
// score) per slot; unfilled slots are zero (score 0 = no face). FaceBoxParams
// above is the single-face decode: one global argmax per channel + hull, which
// merges multiple faces into one box. This one takes local-max candidates per
// channel and groups them into faces by eye-pair hypothesis + canonical-template
// geometry. Consumers address a slot via the `slot` param on CropResample /
// LandmarkOverlay. CPU reference: sdk/demo/face.ts.
export interface FaceBoxesParams extends FaceBoxParams {
  maxFaces: number;   // K — output slots, 1..6
  tol:      number;   // keypoint match radius, × interocular distance (~0.6)
}

// Square crop + resample driven by a box tensor, with per-channel normalization
// folded in ((rgb - mean) / std) — produces the landmark model's input directly.
export interface CropResampleParams {
  outH: number;
  outW: number;
  mean: [number, number, number];
  std:  [number, number, number];
  slot?: number;   // box-tensor slot to crop (default 0) — multi-face
}

export interface LandmarkOverlayParams {
  count:     number;   // landmarks to draw (LandmarkNet: 478)
  thresh:    number;   // hide the overlay when box score < thresh
  pointSize: number;   // dot size in canvas px
  color:     [number, number, number];
  slot?:     number;    // box-tensor slot to read (default 0) — multi-face
  // Draw the image before the dots (default true). Set false to overlay a
  // second/third face's dots onto what a previous overlay already drew — the
  // pass then loads the target instead of clearing it.
  drawImage?: boolean;
}

// Static face-mesh assets for the touch-up effect (ar-scope/build_web_assets.py):
// the canonical 898-triangle mesh expanded to non-indexed vertices — per corner
// a canonical UV (atlas orientation) and the landmark index it samples — plus
// the 512² per-region skin-smoothing weight mask (protection + forehead falloff).
export interface FaceTopology {
  count:      number;         // vertices (898 × 3)
  uv:         Float32Array;   // flat [u0,v0,u1,v1,…], length count·2
  idx:        Float32Array;   // landmark index per vertex, length count
  weightMask: ImageBitmap;    // 512² grayscale (weight in .r)
}

// Smoothing filter applied in UV space — a developer-facing choice:
//   'freq-sep'  : separable gaussian low band + detail·high band. Smooths skin
//                 while keeping pore-level texture; the default.
//   'bilateral' : single edge-preserving pass (range sigma 0.15). Stronger
//                 blemish removal at edges, no detail-keep knob (detail ignored).
export type FaceTouchupStyle = "freq-sep" | "bilateral";

// UV-space skin smoothing: unwrap the face into a canonical UV atlas via the
// landmark mesh, smooth there (pose-independent), composite the smoothed skin
// back through the static weight mask.
export interface FaceTouchupParams {
  strength: number;   // 0..1 — blend of smoothed over original (× weight mask)
  amount:   number;   // smoothing sigma in atlas px
  detail:   number;   // 0..1 — high-frequency keep (freq-sep only)
  thresh:   number;   // skip the effect when box score < thresh (passthrough)
  style?:   FaceTouchupStyle;   // default 'freq-sep'
  // Faces to retouch in one pass — 1 (default) or 4. The K faces share a single
  // atlas as a grid of tiles (1×1 at K=1, 2×2 at K=4), so the blur, the frame
  // copy and the unpack all run ONCE regardless of K; only the two mesh draws
  // scale, via instancing. K=1 is the single-face layout unchanged.
  //
  // Requires `landmarks` to pack K faces' LandmarkNet outputs (ChannelConcat —
  // 478 landmarks each, face f at channel f·956) and `box` to be a 1×K×4 tensor
  // from FaceBoxesFromHeatmaps. Slots with score < thresh are clipped away in
  // the vertex shader, so the cost is flat in the number of faces PRESENT.
  slots?:   number;
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
    // Gather-form transposed conv (flow decoder deconv/upflow). Same flat weight
    // layout as Conv2d — mat4x4[z][o][i], M[in_sub][out_sub] = W(in,out,ky,kx).
    ConvTranspose2d: (input: Tensor, weights: Conv2DWeights,    params: ConvTranspose2dParams) => Op;
    DepthwiseConv2d: (input: Tensor, weights: DepthwiseWeights, params: DepthwiseParams) => Op;
    Add:             (a: Tensor, b: Tensor) => Op;
    Sigmoid:         (input: Tensor) => Op;
    BilinearUpsample:(input: Tensor, params: UpsampleParams) => Op;
    // Top-left crop to (outH, outW) — for flow-decoder skip alignment (training
    // crop_like). outH/outW must be <= the input dims.
    Crop:            (input: Tensor, params: UpsampleParams) => Op;
    BicubicUpsample: (input: Tensor, params: UpsampleParams) => Op;
    ChannelConcat:   (a: Tensor, b: Tensor) => Op;

    // Generic elementwise primitives (temporal models / effects)
    Tanh:            (input: Tensor) => Op;
    ElementwiseMul:  (a: Tensor, b: Tensor) => Op;

    // Optical-flow temporal. Bilinear gather-warp: out[p] = sample(source,
    // p + flowScale·flow[p].xy), clamped to edge. Source + flow share resolution.
    // Used by frame-warp propagation and the stabilizer's warped reference.
    Warp:            (source: Tensor, flow: Tensor, params: WarpParams) => Op;

    // Face-box decode: 5-keypoint heatmaps → 1×1×4 box tensor
    // (cx, cy, halfSide, score), all in FRAME FRACTIONS (halfSide as a fraction
    // of frame WIDTH; the box is square in pixels — the op squares it using the
    // heatmap grid's aspect, which matches the frame's). score = min keypoint
    // peak, 0 if any peak < thresh. Everything stays GPU-resident: CropResample
    // and LandmarkOverlay consume this tensor without a readback.
    FaceBoxFromHeatmaps: (heatmaps: Tensor, params: FaceBoxParams) => Op;

    // Multi-face decode: same heatmaps → 1×K×4 boxes (K = params.maxFaces), one
    // face per slot, score-0 slots meaning "no face". Same units/conventions as
    // FaceBoxFromHeatmaps, so consumers differ only by the `slot` param.
    FaceBoxesFromHeatmaps: (heatmaps: Tensor, params: FaceBoxesParams) => Op;

    // UV-space face touch-up as a Tensor→Tensor EFFECT STAGE: renders the
    // retouched frame into an output tensor the (single, terminal) background
    // compositor consumes as its foreground image — effects compose upstream
    // of the one compositor instead of competing for the canvas. Same passes
    // as presenters.FaceTouchup (which remains for standalone/probe use).
    FaceTouchupStage: (frame: Tensor, landmarks: Tensor, box: Tensor, topo: FaceTopology, params: FaceTouchupParams) => Op;

    // Square crop from `frame` at the box tensor's location, bilinearly
    // resampled to (outH, outW) and normalized ((rgb - mean)/std) — feeds
    // LandmarkNet (ImageNet stats). Mirrors landmark training's warpAffine
    // sampling (src = center + (u - out/2) · side/out).
    CropResample: (frame: Tensor, box: Tensor, params: CropResampleParams) => Op;

    // Flow-gated temporal stabilizer. Per pixel:
    //   env = max(|flow.xy|, release·envPrev.y)                  (peak-hold)
    //   g   = max(clamp((env - tLo)/(tHi - tLo), 0, 1), leak)
    //   out = vec4((g·pred + (1-g)·ref).x, env, 0, 0)
    // Thread the output back as next frame's envPrev (copyTensor) — .y is env.
    Stabilize:       (flow: Tensor, pred: Tensor, ref: Tensor, envPrev: Tensor, params: StabilizeParams) => Op;

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

  // Render-to-display ops. Produce no Tensor — write directly to a canvas.
  // The optional `target` selects which attached canvas to write to (default
  // 'main'); see RenderTarget + attachCanvas. WebGPU honors it; WebGL ignores
  // it (always its single canvas).
  presenters: {
    CompositeSolid:          (image: Tensor, alpha: Tensor, bgColor: [number, number, number], target?: RenderTarget) => Presenter;
    CompositeImage:          (image: Tensor, alpha: Tensor, bg: Tensor, target?: RenderTarget) => Presenter;
    // Same as CompositeImage but bg may be smaller than (image, alpha) — bg
    // is bilinearly sampled. Used by CompositorBlur to absorb the final
    // pyramid upsample into this pass for free.
    CompositeImageBilinear:  (image: Tensor, alpha: Tensor, bg: Tensor, target?: RenderTarget) => Presenter;
    // Passthrough: writes image directly to canvas; no alpha, no bg. Used
    // by RenderOp when the renderer is disabled (true GPU-level passthrough
    // — input frame in, same frame on the canvas).
    CompositePassthrough:    (image: Tensor, target?: RenderTarget) => Presenter;
    // Vertex-pulled landmark dots over the image — the vertex shader reads the
    // landmark tensor (LandmarkNet output, [0,1] crop coords) and the box
    // tensor directly on the GPU and transforms crop → canvas. Hidden when
    // box score < thresh. The no-readback landmark visualization/consumption
    // pattern (the touch-up mesh warp consumes the same buffer the same way).
    LandmarkOverlay:         (image: Tensor, landmarks: Tensor, box: Tensor, params: LandmarkOverlayParams, target?: RenderTarget) => Presenter;
    // UV-space face touch-up: unwrap (mesh draw, vertex-pulled landmarks) →
    // separable gaussian + freq-sep recombine on the atlas → composite the
    // smoothed skin over the frame through the static weight mask. Five GPU
    // passes, zero readback; passthrough when box score < params.thresh.
    FaceTouchup:             (frame: Tensor, landmarks: Tensor, box: Tensor, topo: FaceTopology, params: FaceTouchupParams, target?: RenderTarget) => Presenter;
  };

  // Register an additional output canvas under `name` so presenters can target
  // it. WebGPU configures a second GPUCanvasContext on the shared device (same
  // format, premultiplied alpha) — render passes to distinct canvases share
  // the device's buffers (the alpha tensor) with no readback. WebGL THROWS: a
  // single GL context can't drive two canvases and cross-context texture
  // sharing isn't a thing, so the renderer presents the preview by snapshotting
  // the main canvas instead. Used by the preview-effect feature.
  attachCanvas(name: RenderTarget, canvas: HTMLCanvasElement | OffscreenCanvas): void;

  // Read tensor data back to host as fp32. The tensor must have been allocated
  // by this backend; conversion from fp16 storage is handled internally.
  readback(tensor: Tensor): Promise<Float32Array>;

  // GPU-resident copy of src's contents into dst (identical shape + dtype).
  // Stays entirely on-device — no readback, no CPU round-trip. Used to thread
  // recurrent state across frames: after a model run, copyTensor(model.
  // hiddenState, hPrev) carries the ConvGRU hidden (.z) into the buffer the
  // GRU samples next frame. WebGPU = copyBufferToBuffer; WebGL = copyTexSubImage2D.
  copyTensor(src: Tensor, dst: Tensor): void;

  // Wait for all pending GPU work to complete. Cheaper than readback when
  // you only need a sync barrier (e.g., timing benchmarks). WebGPU uses
  // queue.onSubmittedWorkDone(); WebGL2 uses fenceSync + clientWaitSync
  // or gl.finish() as fallback.
  sync(): Promise<void>;

  destroy(): void;
}
