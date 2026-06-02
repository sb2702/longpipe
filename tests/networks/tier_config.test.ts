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

  it('canvasRes / baseRes are 16:9 and canvas ≥ base', () => {
    for (const [name, cfg] of Object.entries(TIER_CONFIG)) {
      expect(Math.abs(cfg.canvasRes.w / cfg.canvasRes.h - 16 / 9), `${name} canvas`).toBeLessThan(0.02)
      expect(Math.abs(cfg.baseRes.w   / cfg.baseRes.h   - 16 / 9), `${name} base`).toBeLessThan(0.02)
      expect(cfg.canvasRes.w, `${name} canvas≥base`).toBeGreaterThanOrEqual(cfg.baseRes.w)
    }
  })
})
