// Turns a caller's preset choice into a concrete ManualPreset, then
// reconciles the dtype if the backend ended up downgrading (e.g., asked
// for f16 but no shader-f16 → backend forced f32; the preset has to
// follow so weights load and ops dispatch with matching width).
//
//   'auto'                 → microbench, pick largest fitting preset
//   'fast' / 'balanced' / 'quality' → lookup in NAMED_PRESET_INDEX
//   ManualPreset           → passthrough
//
// In all three cases, if resolvedDtype differs from preset.dtype, we
// return a copy with dtype patched.

import type { Backend, Dtype } from '~/model/backend'
import { resolveNamedPreset, type ManualPreset, type PresetName } from '../presets'
import { autotunePreset } from './autotune'

export async function resolvePreset(
  preset:        PresetName | ManualPreset,
  resolvedDtype: Dtype,
  backend:       Backend,
): Promise<ManualPreset> {
  let resolved: ManualPreset

  if (preset === 'auto') {
    resolved = await autotunePreset(backend)
  } else if (typeof preset === 'string') {
    const named = resolveNamedPreset(preset)
    if (!named) throw new Error(`resolve_preset: unknown preset name '${preset}'`)
    resolved = named
  } else {
    resolved = preset
  }

  if (resolved.dtype !== resolvedDtype) {
    return { ...resolved, dtype: resolvedDtype }
  }
  return resolved
}
