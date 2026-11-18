// Which segmentation model the pipeline runs.
//
//   'longpipe'          (default) Longpipe's own hydranet tiers, xs–xl: matting +
//                       face + optical-flow heads off one encoder. Presets,
//                       autotune and the adaptive controller pick the tier.
//   'mediapipe-selfie'  Google's MediaPipe Selfie Segmenter (Apache-2.0 weights)
//                       on the same backends / transport / compositor. Background
//                       effects only. Variant picked from the output aspect:
//                       16:9-ish → landscape (256×144), otherwise square (256×256).
//   'mediapipe-selfie-square' | 'mediapipe-selfie-landscape'
//                       force a variant.
//
// The MediaPipe model has no face or flow heads, so touch-up, auto-reframe and
// flow-based temporal stabilization need model: 'longpipe'. It is one fixed
// model: presets / autotune / adaptive never swap it (or swap to it).
import type { MediaPipeSelfieVariant } from '~/model/networks/mediapipe_selfie.ts'
import { pickMediaPipeSelfieVariant } from '~/model/networks/mediapipe_selfie.ts'
import { LongpipeError } from './errors'

export type ModelChoice =
  | 'longpipe'
  | 'mediapipe-selfie'
  | 'mediapipe-selfie-square'
  | 'mediapipe-selfie-landscape'

// What the worker is told to build. Resolved on the main thread (needs the
// output size for the variant pick) and sent in InitData.
export type ResolvedModel =
  | { family: 'longpipe' }
  | { family: 'mediapipe-selfie'; variant: MediaPipeSelfieVariant }

const CHOICES: readonly ModelChoice[] = ['longpipe', 'mediapipe-selfie', 'mediapipe-selfie-square', 'mediapipe-selfie-landscape']

export const isMediaPipe = (m: ResolvedModel): m is Extract<ResolvedModel, { family: 'mediapipe-selfie' }> =>
  m.family === 'mediapipe-selfie'

export function resolveModel(choice: ModelChoice | undefined, outputSize: { w: number; h: number }): ResolvedModel {
  const c = choice ?? 'longpipe'
  if (!CHOICES.includes(c)) throw new Error(`longpipe: unknown model '${c}' (expected ${CHOICES.map(x => `'${x}'`).join(' | ')})`)
  if (c === 'longpipe') return { family: 'longpipe' }
  const variant: MediaPipeSelfieVariant =
    c === 'mediapipe-selfie-square'    ? 'square'
    : c === 'mediapipe-selfie-landscape' ? 'landscape'
    : pickMediaPipeSelfieVariant(outputSize.w, outputSize.h)
  return { family: 'mediapipe-selfie', variant }
}

// The error every Longpipe-only feature throws under a MediaPipe model:
// 'control-failed' from a control call, 'init-failed' from the constructor.
export function requiresLongpipe(feature: string, model: string, type: 'control-failed' | 'init-failed' = 'control-failed'): LongpipeError {
  return new LongpipeError(type,
    `longpipe: ${feature} requires a Longpipe model; set model: 'longpipe' ` +
    `(model '${model}' is segmentation-only — background effects only)`,
  )
}

// Construction-time checks for options that only make sense with Longpipe tiers.
// Throws synchronously from `new EffectsPipeline()`.
export function validateModelOptions(choice: ModelChoice | undefined, opts: {
  touchup?: unknown
  reframe?: unknown
  preset?:  unknown
}): void {
  const model = choice ?? 'longpipe'
  if (model === 'longpipe') return
  if (opts.touchup) throw requiresLongpipe('touchup', model, 'init-failed')
  if (opts.reframe) throw requiresLongpipe('reframe', model, 'init-failed')
  // 'auto' (the default) is accepted and ignored: there is nothing to tune.
  if (opts.preset !== undefined && opts.preset !== 'auto')
    throw requiresLongpipe(`preset ${JSON.stringify(opts.preset)}`, model, 'init-failed')
}
