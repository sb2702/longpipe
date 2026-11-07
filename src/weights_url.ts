// Default base URL for every fetched asset (video model weights, landmark
// assets, audio denoise wasm + weights). Versioned in the path so SDK upgrades
// that change weight shapes can move to a new prefix without breaking older
// SDKs in the wild. One versioned path hosts everything — filenames
// disambiguate; no per-kind subfolder.
//
// Lives in its own module (not pipeline/index.ts) so standalone entry points
// like `longpipe/audio` can default to it without importing the pipeline.
export const DEFAULT_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/v/0.0.5/'
