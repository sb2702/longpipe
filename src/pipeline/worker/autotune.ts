// Microbench at init for `preset: 'auto'`. Probe in increasing-size order;
// pick the largest preset whose model time fits the FPS budget.
//
// We don't need real weights — GPU work is shape-driven, so zero-filled
// buffers of the correct sizes give the same timing as real weights. The
// `synthBackend` wrapper intercepts weight-taking ops (Conv2d / etc.) and
// allocates synthesized buffers based on op params + input.c, ignoring
// whatever weights tree the network passes in. The network class is
// constructed with a recursive Proxy as ModelWeights — it navigates
// w.encoder.s1[0].expand.weights without throwing; the synth wrapper
// ignores the value at the leaf. No changes needed to ops or networks.

import type {
  Backend, Tensor,
  Conv2dParams, DepthwiseParams, UpsampleConv1x1Params,
  ProjResidualParams, ConcatConv2dParams,
} from '~/model/backend.ts'
import type { Conv2DWeights, DepthwiseWeights, ModelWeights } from '~/model/weights.ts'
import { PRESETS, type ManualPreset } from '../presets'
import { TIER_CONFIG } from '~/model/tier_config.ts'
import {WebGLBackend} from "~/model/backends/webgl/index.ts";

// Frame budget = (1000 / sourceFps) × SAFETY_MARGIN. The model gets at
// most this fraction of each input frame; the rest is reserved for
// compositor + transport + main-thread headroom. 0.5 = "model takes at
// most half the frame", which keeps xl/large from being picked just
// because they squeak under the 33ms wall.
const SAFETY_MARGIN = 0.5
const WARMUP_ITERS  = 3
const TIMED_ITERS   = 10
const DEFAULT_SOURCE_FPS = 30

import { createLogger } from '../debug'
const log = createLogger('autotune')

type RunnableCtor = new (b: Backend, i: Tensor, w: ModelWeights) => { run(): void }

// NOTE: this benches the BASE network alone (encoder+decoder) at base res —
// the dominant cost and a reasonable proxy for relative tier ordering. It does
// NOT model the wrapper down/up-path (which at xl runs at 1280×768 and is not
// negligible) or the ConvGRU. Adaptive FPS-downgrade corrects an over-pick at
// runtime. Follow-up: bench the full TierModel (needs synth weights for the
// wrapper/gru ops, beyond the 4 the synth backend intercepts today).

export async function autotunePreset(
  backend:         Backend,
  sourceFpsTarget: number = DEFAULT_SOURCE_FPS,
): Promise<ManualPreset> {

  // Be conservative with WebGL (heavier fence-sync, no upgrade path): half the
  // budget. Local — never mutate the module-level constant (would compound
  // across repeated inits).
  const safetyMargin = backend instanceof WebGLBackend ? SAFETY_MARGIN * 0.5 : SAFETY_MARGIN
  const budgetMs = (1000 / sourceFpsTarget) * safetyMargin
  log(`start; budget per source frame: ${budgetMs.toFixed(1)}ms (source ${sourceFpsTarget}fps × ${safetyMargin} safety)`)
  log('backend dtype:', backend.dtype)

  let best: ManualPreset | null = null
  for (const preset of PRESETS) {
    // 'auto' tops out at LARGE — the reference tier. xl is the explicit
    // opt-in flagship ('quality' named preset or a manual preset); auto
    // should not surprise callers with its cost.
    if (preset.model === 'xl') continue
    if (!TIER_CONFIG[preset.model]) {
      log('skip', preset.model, '(no TIER_CONFIG entry)')
      continue
    }
    try {
      const baseRes = TIER_CONFIG[preset.model].baseRes
      log(`bench ${preset.model} base @ ${baseRes.w}×${baseRes.h} skipFrames=${preset.skipFrames} …`)
      const ms = await microbench(backend, preset)
      const ok = ms <= budgetMs
      log(`  ${preset.model}: ${ms.toFixed(1)}ms / source frame ${ok ? '✓ within budget' : '✗ over budget'}`)
      // Keep the cheapest implementable preset as a floor in case nothing
      // else fits. Once we hit one that's over budget, larger presets at
      // the same skip-rate are also over — break.
      if (ok || best === null) best = preset
      if (!ok) break
    } catch (err) {
      log('  bench failed for', preset.model, ':', err)
      break
    }
  }

  if (!best) throw new Error('autotune: no implementable preset available')
  log('picked', best.model)
  return best
}

// ── Synth backend ───────────────────────────────────────────────────────────
// Wraps real backend.ops to allocate zero-filled weights of the right
// shape per op call. Other ops (Add/Sigmoid/Input/etc.) and methods
// (tensor/upload/readback) pass through unchanged via spread.

function synthBackend(backend: Backend): Backend {
  // Object.create rather than spread: backend is a class instance, so
  // methods (tensor / upload / sync / readback) live on the prototype.
  // Spread would only copy own properties, leaving the wrapper without
  // those methods. Object.create preserves the prototype chain.
  const wrapped = Object.create(backend) as Backend
  wrapped.ops = {
    ...backend.ops,
      Conv2d: (input, _w, params: Conv2dParams) => {
        const synth: Conv2DWeights = {
          weights: new Float32Array(params.kernel * params.kernel * input.c * params.outChannels),
          bias:    new Float32Array(params.outChannels),
        }
        return backend.ops.Conv2d(input, synth, params)
      },
      DepthwiseConv2d: (input, _w, params: DepthwiseParams) => {
        const synth: DepthwiseWeights = {
          weights: new Float32Array(params.kernel * params.kernel * input.c),
          bias:    new Float32Array(input.c),
        }
        return backend.ops.DepthwiseConv2d(input, synth, params)
      },
      Conv2dAdd: (input, skip, _w, params: Conv2dParams) => {
        const synth: Conv2DWeights = {
          weights: new Float32Array(params.kernel * params.kernel * input.c * params.outChannels),
          bias:    new Float32Array(params.outChannels),
        }
        return backend.ops.Conv2dAdd(input, skip, synth, params)
      },
      UpsampleConv1x1: (input, _w, params: UpsampleConv1x1Params) => {
        const synth: Conv2DWeights = {
          weights: new Float32Array(input.c * params.outChannels),
          bias:    new Float32Array(params.outChannels),
        }
        return backend.ops.UpsampleConv1x1(input, synth, params)
      },
      // ProjResidual: 1×1 conv (input.c → outChannels) + residual add.
      ProjResidual: (input, skip, _w, params: ProjResidualParams) => {
        const synth: Conv2DWeights = {
          weights: new Float32Array(input.c * params.outChannels),
          bias:    new Float32Array(params.outChannels),
        }
        return backend.ops.ProjResidual(input, skip, synth, params)
      },
      // ConcatConv2d: 3×3 conv over concat(a, b) → outChannels (decoder conv1).
      ConcatConv2d: (a, b, _w, params: ConcatConv2dParams) => {
        const synth: Conv2DWeights = {
          weights: new Float32Array(9 * (a.c + b.c) * params.outChannels),
          bias:    new Float32Array(params.outChannels),
        }
        return backend.ops.ConcatConv2d(a, b, synth, params)
      },
    }
  return wrapped
}

// Recursive Proxy that returns itself on any property/index access. Lets
// the network constructor walk w.encoder.s1[0].expand.weights without
// throwing; the synth backend wrapper above ignores the value anyway.
function dummyWeights(): ModelWeights {
  const p: unknown = new Proxy({}, { get: () => p })
  return p as ModelWeights
}

// ── Bench loop ──────────────────────────────────────────────────────────────

async function microbench(backend: Backend, preset: ManualPreset): Promise<number> {
  const cfg = TIER_CONFIG[preset.model]
  if (!cfg) throw new Error(`microbench: no TIER_CONFIG entry for '${preset.model}'`)
  const Ctor = cfg.base as unknown as RunnableCtor

  const sb      = synthBackend(backend)
  const input   = sb.tensor(cfg.baseRes.h, cfg.baseRes.w, 4)
  const network = new Ctor(sb, input, dummyWeights())

  // Warmup — first iters include shader compile + pipeline cache fill;
  // keep these out of the timed window. Sync each so subsequent iters
  // start fresh.
  for (let i = 0; i < WARMUP_ITERS; i++) {
    network.run()
    await backend.sync()
  }

  // Batched timed window: dispatch N runs back-to-back, sync once at the
  // end, divide by N. Avoids per-iter sync overhead skewing small-model
  // numbers; backend's queue serialises dispatches so total wall time is
  // a clean sum of per-run cost.
  // Bench loop simulates `TIMED_ITERS` source-frame ticks. The model only
  // runs on 1 of every (skipFrames + 1) ticks, matching the renderer's
  // actual skip pattern. Returns avg cost per source frame, so a preset
  // that runs the model every other frame is fairly compared against one
  // that runs every frame.
  const stride = preset.skipFrames + 1
  const t0 = performance.now()
  for (let i = 0; i < TIMED_ITERS; i++) {
    if (i % stride === 0) network.run()
  }
  await backend.sync()
  const total = performance.now() - t0
  return total / TIMED_ITERS
}
