// Transport-agnostic core. Wraps RenderOp from sdk/src/model/render_op.ts
// with frame-skipping logic. Adapters call process(frame) per input frame;
// renderer decides whether to run the model this frame.

import type { Backend } from '~/model/backend'
import type { ManualPreset } from '../presets'
import type { EffectConfig } from '../effects'
import type { RendererStats } from '../messages'

export interface RendererOptions {
  backend: Backend
  canvas:  OffscreenCanvas
  preset:  ManualPreset
  weights: ArrayBuffer
  effect:  EffectConfig
  enabled: boolean
}

export class Renderer {
  readonly canvas: OffscreenCanvas

  private preset:  ManualPreset
  private enabled: boolean

  // Frame-skipping: model runs at preset.modelFps; display always runs.
  private lastModelRunUs: number = -Infinity

  // Stats — rolling windows
  private framesRenderedAt: number[] = []
  private modelRunsMs:      number[] = []
  private skippedCount:     number   = 0

  constructor(opts: RendererOptions) {
    this.canvas  = opts.canvas
    this.preset  = opts.preset
    this.enabled = opts.enabled
    // TODO:
    //  - parse weights via loadWeightsFromBinary
    //  - pick network class for opts.preset.model
    //  - construct Input op for network input + RenderOp with effect config
  }

  // Per-frame entry point. Always cheap (display). Sometimes expensive (model).
  process(_frame: VideoFrame): void {
    // TODO:
    //  if (!this.enabled) { passthrough — blit frame to canvas; return }
    //  this.renderOp.setSource(frame)
    //  if (this.shouldRunModel(frame.timestamp)) {
    //    const t = performance.now()
    //    this.renderOp.runModel()
    //    this.modelRunsMs.push(performance.now() - t)
    //  } else { this.skippedCount++ }
    //  this.renderOp.runDisplay()
    //  this.framesRenderedAt.push(performance.now())
  }

  private shouldRunModel(timestampUs: number): boolean {
    const intervalUs = 1_000_000 / this.preset.modelFps
    if (timestampUs - this.lastModelRunUs >= intervalUs) {
      this.lastModelRunUs = timestampUs
      return true
    }
    return false
  }

  setEffect(_config: EffectConfig): void {
    // TODO: translate ImageBitmap → Tensor if needed; this.renderOp.setBackground(...)
  }

  setEnabled(on: boolean): void {
    this.enabled = on
  }

  // Runtime preset swap (used by adaptive controller in v0.2). Renderer
  // is designed to support this from day one — no constructor-frozen state.
  setPreset(_preset: ManualPreset, _weights: ArrayBuffer): void {
    // TODO:
    //  - construct new network for preset.model with weights
    //  - rebuild RenderOp around new network (preserve effect config)
    //  - swap atomically on next process() call
    //  - update this.preset
  }

  getStats(): RendererStats {
    // TODO: trim windows to last ~1s, compute medians
    return {
      fps:     0,
      modelMs: 0,
      skipped: this.skippedCount,
    }
  }
}
