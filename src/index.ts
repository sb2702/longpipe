// Public surface — `Pipeline` internally, exported as `EffectsPipeline` so
// the import site reads as what it does (`import { EffectsPipeline } from
// 'longpipe'`). Internal code keeps the shorter `Pipeline` name.
export { Pipeline as EffectsPipeline } from '~/pipeline/index.ts'
export type {
  PipelineOptions, TouchupOptions,
  BackgroundInput, Background, BlurInput, ColorInput, ImageInput, VideoInput,
  PresetName, ManualPreset, ModelName,
  AudioMode, AudioInput, DenoiseOptions, DenoiseModel, DenoiseTier, DenoiseModelOption, AudioStats,
  PipelineError, ErrorSource,
} from '~/pipeline/index.ts'

// Lower-level types for callers building on top of the backend directly.
export type { Backend, Tensor, Op, Activation, Conv2dParams, DepthwiseParams, FaceTouchupStyle } from '~/model/backend.ts'
export { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
