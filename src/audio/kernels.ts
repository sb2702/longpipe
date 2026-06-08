// Audio-denoise kernel registry. Each kernel is a bare ("glue-free") wasm that
// runs one 480-sample (10 ms @ 48 kHz) hop at a time with internal streaming
// state — instantiated on the audio render thread (no fetch/ESM there), driven
// through linear memory. Mirrors the video model registry, but for audio.
//
// Three production models back three tiers:
//   - dfn     (tract-free DFN3 f32 "lite", ~250 KB wasm + weights pack)  → high
//   - dfnint8 (DFN3 with int8 GRUs, smaller + faster on weak HW)         → mid
//   - rnnoise (classic 2017, weights embedded, ~134 KB, no SIMD needed)  → low
// See HANDOFF.md (audio-denoising repo) §2/§9 for why these three.

export type DenoiseModel = 'rnnoise' | 'dfn' | 'dfnint8'
export type DenoiseTier = 'high' | 'mid' | 'low'
// What the caller may pass for `model`: 'auto' (probe picks), a tier shorthand,
// or an explicit model — exactly like the video `preset` accepts named or manual.
export type DenoiseModelOption = 'auto' | DenoiseTier | DenoiseModel

export const HOP = 480
export const SAMPLE_RATE = 48000

export interface KernelSpec {
  // Asset filenames, resolved against the audio weights base URL at fetch time.
  wasm:    string
  // Weights pack filename, or null when the model bakes weights into the wasm
  // (rnnoise). DFN models fetch this separately (~13.8 MB f32 / 7.9 MB int8).
  weights: string | null
  // Sample scaling at the wasm boundary: rnnoise wants int16-magnitude
  // ([-32768, 32768]); DFN wants normalized [-1, 1].
  scaleIn:  number
  scaleOut: number
  // Whether the kernel requires wasm SIMD (DFN does; rnnoise has a scalar path).
  needsSimd: boolean
  // Bare-ABI export names the processor calls. `create` builds streaming state;
  // for DFN it takes the uploaded weights, for rnnoise it's parameterless.
  exports: {
    malloc:      string
    create:      string   // (weightsPtr, weightsLen) -> statePtr | () -> statePtr
    process:     string   // (statePtr, inPtr, outPtr) -> number
    setBeta?:    string   // (statePtr, beta)   — DFN only
    setGruLeak?: string   // (statePtr, factor) — DFN only
  }
}

export const KERNELS: Record<DenoiseModel, KernelSpec> = {
  rnnoise: {
    wasm: 'rnnoise.wasm', weights: null,
    scaleIn: 32768, scaleOut: 1 / 32768, needsSimd: false,
    exports: { malloc: 'rn_malloc', create: 'rn_create', process: 'rn_process' },
  },
  dfn: {
    wasm: 'dfn.wasm', weights: 'dfn_weights.pack',
    scaleIn: 1, scaleOut: 1, needsSimd: true,
    exports: {
      malloc: 'df_lite_malloc', create: 'df_lite_create', process: 'df_lite_process',
      setBeta: 'df_lite_set_beta', setGruLeak: 'df_lite_set_gru_leak',
    },
  },
  dfnint8: {
    wasm: 'dfnint8.wasm', weights: 'dfn_weights_int8.pack',
    scaleIn: 1, scaleOut: 1, needsSimd: true,
    exports: {
      malloc: 'df_lite_malloc', create: 'df_lite_create', process: 'df_lite_process',
      setBeta: 'df_lite_set_beta', setGruLeak: 'df_lite_set_gru_leak',
    },
  },
}

// Tier → model. high = best quality (DFN f32), mid = DFN int8 (smaller/faster),
// low = RNNoise floor (also the only no-SIMD option).
export const TIER_MODEL: Record<DenoiseTier, DenoiseModel> = {
  high: 'dfn',
  mid:  'dfnint8',
  low:  'rnnoise',
}

const TIERS: readonly DenoiseTier[] = ['high', 'mid', 'low']
const MODELS: readonly DenoiseModel[] = ['rnnoise', 'dfn', 'dfnint8']

export function isTier(x: string): x is DenoiseTier { return (TIERS as readonly string[]).includes(x) }
export function isModel(x: string): x is DenoiseModel { return (MODELS as readonly string[]).includes(x) }

// Resolve a caller option to a concrete model. 'auto' is NOT resolved here —
// the probe (probe.ts) decides it and passes a concrete tier/model back.
export function resolveModel(opt: Exclude<DenoiseModelOption, 'auto'>): DenoiseModel {
  return isTier(opt) ? TIER_MODEL[opt] : opt
}
