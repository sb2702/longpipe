import type { Dtype } from '~/model/backend'

export type ModelName = 'xxs' | 'xs' | 'small' | 'compact' | 'medium' | 'large' | 'xl'

export type PresetName = 'fast' | 'balanced' | 'quality' | 'auto'

export interface ManualPreset {
  model:      ModelName
  dtype:      Dtype
  resolution: { w: number; h: number }
  // Number of input frames the model SKIPS between runs. 0 = run every
  // frame, 1 = every other frame, etc. The compositor still runs every
  // frame using whatever's in the alpha tensor. See docs/MODEL_PLAN.md.
  skipFrames: number
}

// Linear ordering: index 0 = cheapest per source frame, last = most
// expensive. Adaptive switching (v0.2) walks this index up/down.
//
// Per-source-frame cost = per-run cost / (skipFrames + 1). xl/large run
// every frame (skipFrames=0); the others skip every other frame
// (skipFrames=1).
export const PRESETS: ManualPreset[] = [
  // xxs: same architecture as xs (small encoder + standard decoder), runs at
  // 128×72 with skipFrames=3 (one model run per 4 source frames). For very
  // weak hardware. Reuses model_xs.bin since architecture matches; SDK will
  // fetch model_xxs.bin so upload a copy on the CDN.
  { model: 'xxs',     dtype: 'f16', resolution: { w: 128, h: 72  }, skipFrames: 3 },
  { model: 'xs',      dtype: 'f16', resolution: { w: 192, h: 108 }, skipFrames: 1 },
  { model: 'small',   dtype: 'f16', resolution: { w: 256, h: 144 }, skipFrames: 1 },
  { model: 'compact', dtype: 'f16', resolution: { w: 256, h: 144 }, skipFrames: 1 },
  { model: 'medium',  dtype: 'f16', resolution: { w: 256, h: 144 }, skipFrames: 1 },
  { model: 'large',   dtype: 'f32', resolution: { w: 256, h: 144 }, skipFrames: 0 },
  { model: 'xl',      dtype: 'f32', resolution: { w: 512, h: 288 }, skipFrames: 0 },
]

// Named shortcuts → index into PRESETS. 'auto' resolved via microbench at init.
export const NAMED_PRESET_INDEX: Record<Exclude<PresetName, 'auto'>, number> = {
  fast:     1,   // xs (xxs is too aggressive for the 'fast' shortcut)
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
