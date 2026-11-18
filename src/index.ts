// Public surface — `Pipeline` internally, exported as `EffectsPipeline` so
// the import site reads as what it does (`import { EffectsPipeline } from
// 'longpipe'`). Internal code keeps the shorter `Pipeline` name.
export { Pipeline as EffectsPipeline, LongpipeError } from '~/pipeline/index.ts'
// The fixed model tiers, cheapest first (xs, small, medium, large, xl). Pass one
// as `preset` or to `setPreset()` to pin a tier. Read-only.
export { PRESETS } from '~/pipeline/presets.ts'
export type {
  PipelineOptions, TouchupOptions, RecoveredInfo, TrackChange, AudioTrackChange,
  PipelineErrorType, ActiveBackend, BackendPreference, PipelineStats, PipelineStatus, ReadyInfo,
  BackgroundInput, Background, BlurInput, ColorInput, ImageInput, VideoInput,
  PresetName, ManualPreset, ModelName, ModelChoice,
  AudioMode, AudioInput, DenoiseOptions, DenoiseModel, DenoiseTier, DenoiseModelOption, AudioStats,
  PipelineError, ErrorSource,
} from '~/pipeline/index.ts'

// Lower-level types for callers building on top of the backend directly.
export type { Backend, Tensor, Op, Activation, Conv2dParams, DepthwiseParams, FaceTouchupStyle } from '~/model/backend.ts'
export { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
