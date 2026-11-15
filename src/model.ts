// `longpipe/model` — the model layer, usable WITHOUT the pipeline.
//
// Bring your own frame loop, canvas, and transport; drive the tier model
// directly. This is the surface `sdk/demo/main.ts` uses. Documented at
// https://longpipe.dev/docs/model-layer.
//
// This is a SEPARATE ENTRY POINT on purpose. `WebGLBackend` pulls ~114KB of
// GLSL into whatever bundle references it, and the main entry's pipeline
// carries its backends inside the inlined worker — so exporting the model
// layer from 'longpipe' would tax every EffectsPipeline consumer for code
// they never touch. Import from 'longpipe' for the pipeline, 'longpipe/model'
// for this. Nothing here is new code; it's all already in the package.

// ── Backends ──────────────────────────────────────────────────────────────
// Both take { canvas, dtype } and satisfy the same Backend interface.
export { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
export { WebGLBackend }  from '~/model/backends/webgl/index.ts'
export type {
  Backend, Tensor, Op, Presenter, Activation, Dtype, InputOp, ImageSource,
  RenderTarget, Conv2dParams, DepthwiseParams, UpsampleParams, FaceTouchupStyle,
} from '~/model/backend.ts'

// ── Tier model ────────────────────────────────────────────────────────────
// TierModel = base network + U-Net wrapper. TIER_CONFIG says which pieces and
// at what resolutions ("the model is code" — the .bin carries only weights).
export { TierModel } from '~/model/tier_model.ts'
export type { BaseNetwork, BaseNetworkCtor } from '~/model/tier_model.ts'
export { TIER_CONFIG } from '~/model/tier_config.ts'
export type { TierConfig, Res } from '~/model/tier_config.ts'

// ── MediaPipe selfie segmenter (optional, segmentation-only) ─────────────
// Google's selfie segmenter on the same backends / Input op / RenderOp. Its
// output has TierModel's contract (alpha in channel 0), so attachNetwork takes
// either. No face / flow heads: touch-up, reframe and flow stabilization need
// a Longpipe tier.
export {
  MediaPipeSelfie, MEDIAPIPE_SELFIE_CONFIG, pickMediaPipeSelfieVariant,
} from '~/model/networks/mediapipe_selfie.ts'
export type {
  MediaPipeSelfieVariant, MediaPipeSelfieConfig, MediaPipeSelfieWeights,
} from '~/model/networks/mediapipe_selfie.ts'
export { DEFAULT_WEIGHTS_BASE_URL, DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL } from '~/weights_url.ts'

// ── Weights ───────────────────────────────────────────────────────────────
// Parses a model_<tier>.bin (or .f16.bin) into the nested weights object.
export { loadWeightsFromBinary } from '~/utils/loadWeights.ts'
export type {
  ModelWeights, UNetWrapperWeights, FlowWeights, FaceWeights, LandmarkWeights,
  Conv2DWeights, DepthwiseWeights, MBConvWeights, DecoderBlockWeights,
} from '~/model/weights.ts'

// ── Render orchestrator ───────────────────────────────────────────────────
// network → alpha upscale → composite to the backend's canvas. Optional: skip
// it and read the alpha tensor back yourself if you only want a matte.
export { RenderOp } from '~/model/render_op.ts'
export type {
  BackgroundConfig, CompositeSpec, UpscalerMode, RenderOptions,
} from '~/model/render_op.ts'
