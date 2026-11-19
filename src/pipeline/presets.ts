import type { Dtype } from '~/model/backend.ts'

// The five production tiers. Each has a TIER_CONFIG entry (base + wrapper +
// GRU policy) in ~/model/tier_config.ts and a shipped model_<name>.bin.
export type ModelName = 'xs' | 'small' | 'medium' | 'large' | 'xl'

// Shortcuts ('fast' | 'balanced' | 'quality'), 'auto' (benchmark at init), or a
// tier name to pin that tier (the same names ready.preset / getStats() report).
export type PresetName = 'fast' | 'balanced' | 'quality' | 'auto' | ModelName

export interface ManualPreset {
  model:      ModelName
  dtype:      Dtype
  // x_hr (network input) AND alpha-output resolution = the tier's canvas res.
  // MUST match TIER_CONFIG[model].canvasRes (the model layer is authoritative;
  // tests/networks/tier_config.test.ts asserts agreement). The wrapper strides
  // this down to base res internally.
  resolution: { w: number; h: number }
  // Number of input frames the model SKIPS between runs. 0 = run every
  // frame, 1 = every other frame, etc. The compositor still runs every
  // frame using whatever's in the alpha tensor. See docs/MODEL_PLAN.md.
  skipFrames: number
}

// Linear ordering: index 0 = cheapest per source frame, last = most
// expensive. The adaptive controller walks this index up/down at runtime.
//
// Per-source-frame cost = per-run cost / (skipFrames + 1). xl/large run
// every frame (skipFrames=0); small/medium skip every other frame
// (skipFrames=1); xs skips 2 of every 3 (skipFrames=2). resolution = canvas
// res (mirrors TIER_CONFIG.canvasRes).
export const PRESETS: readonly Readonly<ManualPreset>[] = [
  { model: 'xs',      dtype: 'f16', resolution: { w: 384,  h: 240 }, skipFrames: 2 },
  { model: 'small',   dtype: 'f16', resolution: { w: 384,  h: 224 }, skipFrames: 1 },
  { model: 'medium',  dtype: 'f16', resolution: { w: 512,  h: 320 }, skipFrames: 1 },
  { model: 'large',   dtype: 'f32', resolution: { w: 640,  h: 400 }, skipFrames: 0 },
  { model: 'xl',      dtype: 'f32', resolution: { w: 1280, h: 768 }, skipFrames: 0 },
]

// Named shortcuts → index into PRESETS. 'auto' resolved via microbench at init.
export const NAMED_PRESET_INDEX: Record<Exclude<PresetName, 'auto'>, number> = {
  fast:     0,   // xs
  balanced: 2,   // medium
  quality:  4,   // xl
  xs:       0,
  small:    1,
  medium:   2,
  large:    3,
  xl:       4,
}

export function resolveNamedPreset(name: PresetName): ManualPreset | null {
  if (name === 'auto') return null   // caller must run autotune
  const i = Object.hasOwn(NAMED_PRESET_INDEX, name) ? NAMED_PRESET_INDEX[name] : undefined
  return i === undefined ? null : PRESETS[i]
}

export function presetIndex(p: ManualPreset): number {
  return PRESETS.findIndex(x =>
    x.model === p.model &&
    x.dtype === p.dtype &&
    x.resolution.w === p.resolution.w &&
    x.resolution.h === p.resolution.h,
  )
}
