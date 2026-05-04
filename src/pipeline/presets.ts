import type { Dtype } from '~/model/backend'

export type ModelName = 'xs' | 'small2' | 'small' | 'compact' | 'large' | 'xl'

export type PresetName = 'fast' | 'balanced' | 'quality' | 'auto'

export interface ManualPreset {
  model:      ModelName
  dtype:      Dtype
  resolution: { w: number; h: number }
  modelFps:   number
}

// Linear ordering: index 0 = fastest, last = slowest. Adaptive switching
// (v0.2) walks this index up/down. See docs/PIPELINE.md.
export const PRESETS: ManualPreset[] = [
  { model: 'xs',      dtype: 'f16', resolution: { w: 192, h: 108 }, modelFps: 30 },
  { model: 'small2',  dtype: 'f16', resolution: { w: 192, h: 108 }, modelFps: 30 },
  { model: 'small',   dtype: 'f16', resolution: { w: 256, h: 144 }, modelFps: 20 },
  { model: 'compact', dtype: 'f16', resolution: { w: 256, h: 144 }, modelFps: 20 },
  { model: 'large',   dtype: 'f32', resolution: { w: 256, h: 144 }, modelFps: 15 },
  { model: 'xl',      dtype: 'f32', resolution: { w: 512, h: 288 }, modelFps: 15 },
]

// Named shortcuts → index into PRESETS. 'auto' resolved via microbench at init.
export const NAMED_PRESET_INDEX: Record<Exclude<PresetName, 'auto'>, number> = {
  fast:     0,
  balanced: 3,   // compact
  quality:  5,   // xl
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
