import type { Dtype } from '~/model/backend'

export type ModelName = 'xs' | 'small2' | 'small' | 'compact' | 'medium' | 'large' | 'xl'

export type PresetName = 'fast' | 'balanced' | 'quality' | 'auto'

export interface ManualPreset {
  model:      ModelName
  dtype:      Dtype
  resolution: { w: number; h: number }
  modelFps:   number       // per docs/MODEL_PLAN.md: skipFrames=0 → 30; skipFrames=1 → 15
}

// Linear ordering: index 0 = fastest per second, last = slowest per second.
// Adaptive switching (v0.2) walks this index up/down.
//
// Per-second cost = per-frame cost × modelFps. xl/large run every frame
// (modelFps=30) so their per-second cost is ~2× their per-frame cost; the
// other presets skip every other frame (modelFps=15). See docs/MODEL_PLAN.md.
export const PRESETS: ManualPreset[] = [
  { model: 'xs',      dtype: 'f16', resolution: { w: 192, h: 108 }, modelFps: 15 },
  { model: 'small2',  dtype: 'f16', resolution: { w: 192, h: 108 }, modelFps: 15 },
  { model: 'small',   dtype: 'f16', resolution: { w: 256, h: 144 }, modelFps: 15 },
  { model: 'compact', dtype: 'f16', resolution: { w: 256, h: 144 }, modelFps: 15 },
  { model: 'medium',  dtype: 'f16', resolution: { w: 256, h: 144 }, modelFps: 15 },
  { model: 'large',   dtype: 'f32', resolution: { w: 256, h: 144 }, modelFps: 30 },
  { model: 'xl',      dtype: 'f32', resolution: { w: 512, h: 288 }, modelFps: 30 },
]

// Named shortcuts → index into PRESETS. 'auto' resolved via microbench at init.
export const NAMED_PRESET_INDEX: Record<Exclude<PresetName, 'auto'>, number> = {
  fast:     0,    // xs
  balanced: 4,   // medium
  quality:  6,   // xl
}

export function resolveNamedPreset(name: PresetName): ManualPreset | null {
  if (name === 'auto') return null   // caller must run autotune
  return PRESETS[NAMED_PRESET_INDEX[name]]
}

export function presetIndex(p: ManualPreset): number {
  return PRESETS.findIndex(x =>
    x.model === p.model &&
    x.dtype === p.dtype &&
    x.resolution.w === p.resolution.w &&
    x.resolution.h === p.resolution.h,
  )
}
