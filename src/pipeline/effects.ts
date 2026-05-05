// Re-exports the public effect-configuration types. Currently `background`
// is the only effect; future top-level effects (touchup, lighting, …) will
// each get their own module parallel to background.ts and be re-exported
// from here. The previous `EffectConfig` wrapper has been removed —
// effects are top-level keys on PipelineOptions, not nested under an
// `effect:` discriminator.

export type {
  BackgroundInput,
  Background,
  BlurInput,
  ImageInput,
  VideoInput,
} from './background'
