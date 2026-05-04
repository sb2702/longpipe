// Public-facing effect configuration. Translated to renderer-internal
// types (which use Tensors instead of ImageBitmaps) inside the worker.
// Shared between main + worker; lives in its own file to keep messages.ts
// and pipeline.ts from forming a circular import.

export type EffectConfig =
  | { effect: 'background'; config: BackgroundConfig }
  // | { effect: 'touchup';    config: TouchupConfig }    // future

export type BackgroundConfig =
  | { blur:  true | { sigma: number } }
  | { image: ImageBitmap | VideoFrame }
  | { color: [number, number, number] }
