import { describe, it, expect } from 'vitest'
import { TIER_CONFIG } from '~/model/tier_config'
import { PRESETS, NAMED_PRESET_INDEX, resolveNamedPreset } from '~/pipeline/presets'

// Guards the presets ↔ tier_config seam. presets.ts hardcodes per-tier canvas
// resolutions (to avoid bundling the network classes into the main thread);
// tier_config.ts is the authoritative model-layer source. These must agree, or
// the renderer (which reads tier_config) and autotune/labels (which read
// presets) diverge silently.
describe('tier_config ↔ presets', () => {
  it('every preset model has a TIER_CONFIG entry', () => {
    for (const p of PRESETS) expect(TIER_CONFIG[p.model], p.model).toBeDefined()
  })

  it('preset.resolution matches TIER_CONFIG.canvasRes', () => {
    for (const p of PRESETS) {
      const c = TIER_CONFIG[p.model].canvasRes
      expect(p.resolution, p.model).toEqual({ w: c.w, h: c.h })
    }
  })

  it('named shortcuts resolve to a real preset', () => {
    for (const name of Object.keys(NAMED_PRESET_INDEX) as Array<keyof typeof NAMED_PRESET_INDEX>) {
      const preset = resolveNamedPreset(name)
      expect(preset, name).not.toBeNull()
      expect(TIER_CONFIG[preset!.model], name).toBeDefined()
    }
  })

  // Base shapes mirror train_run.py PRESETS 'landscape' (not exactly 16:9 —
  // e.g. large is 256×160). The structural invariant is the wrapper stride:
  // canvasRes = baseRes × canvas_mul for the tier's variant.
  it('canvasRes = baseRes × wrapper canvas_mul', () => {
    const CANVAS_MUL: Record<string, number> = { A: 2, B: 3, D: 2.5, E: 4 }
    for (const [name, cfg] of Object.entries(TIER_CONFIG)) {
      const mul = CANVAS_MUL[cfg.wrapper.variant]
      expect(mul, `${name} known variant`).toBeDefined()
      expect(cfg.canvasRes.w, `${name} canvas w`).toBe(Math.round(cfg.baseRes.w * mul))
      expect(cfg.canvasRes.h, `${name} canvas h`).toBe(Math.round(cfg.baseRes.h * mul))
    }
  })
})
