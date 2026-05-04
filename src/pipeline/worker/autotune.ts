// Microbench at init for `preset: 'auto'`. Probe in increasing-size order;
// pick the largest preset whose model time fits the FPS budget. Caches
// result to localStorage so subsequent sessions skip the probe.
//
// We don't need real weights — GPU work is shape-driven, so constants-fill
// gives the same timing as real weights.

import type { Backend } from '~/model/backend'
import { PRESETS, type ManualPreset } from '../presets'

const SAFETY_MARGIN = 0.7
const WARMUP_ITERS  = 10
const TIMED_ITERS   = 20

export async function autotunePreset(
  backend:        Backend,
  modelFpsTarget: number = 15,
): Promise<ManualPreset> {
  const cached = readCache(backend)
  if (cached) return cached

  const budgetMs = (1000 / modelFpsTarget) * SAFETY_MARGIN
  let best: ManualPreset = PRESETS[0]

  for (const preset of PRESETS) {
    const ms = await microbench(backend, preset)
    if (ms <= budgetMs) best = preset
    else break       // monotonic — once we miss budget, larger presets miss too
  }

  writeCache(backend, best)
  return best
}

async function microbench(_backend: Backend, _preset: ManualPreset): Promise<number> {
  // TODO:
  //  - construct preset's network with constants-fill weights
  //  - allocate input tensor (zero-filled)
  //  - WARMUP_ITERS untimed runs (await GPU sync each)
  //  - TIMED_ITERS timed runs (await GPU sync each); return median ms
  //  - tear down to release GPU memory
  void WARMUP_ITERS; void TIMED_ITERS
  return 0
}

function cacheKey(backend: Backend): string {
  // TODO: include backend type, dtype, GPU info hash (adapter info on WebGPU)
  return `longpipe:autotune:${backend.dtype}`
}

function readCache(backend: Backend): ManualPreset | null {
  try {
    const raw = localStorage.getItem(cacheKey(backend))
    return raw ? JSON.parse(raw) as ManualPreset : null
  } catch { return null }
}

function writeCache(backend: Backend, preset: ManualPreset): void {
  try {
    localStorage.setItem(cacheKey(backend), JSON.stringify(preset))
  } catch { /* ignore quota / private mode */ }
}
