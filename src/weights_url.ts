// Default base URL for every fetched asset (video model weights, landmark
// assets, audio denoise wasm + weights). Versioned in the path so SDK upgrades
// that change weight shapes can move to a new prefix without breaking older
// SDKs in the wild. One versioned path hosts everything — filenames
// disambiguate; no per-kind subfolder.
//
// Lives in its own module (not pipeline/index.ts) so standalone entry points
// like `longpipe/audio` can default to it without importing the pipeline.
export const DEFAULT_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/v/0.0.5/'

// Third-party MediaPipe selfie segmenter weights (Apache-2.0, converted by
// scripts/convert_mediapipe_selfie.py). Versioned separately from Longpipe's own
// weights — they change only if Google's source model does. Used when the caller
// did NOT set weightsBaseUrl; a self-hoster's weightsBaseUrl is used for these
// files too (same flat folder, filenames disambiguate).
export const DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL = 'https://cdn.longpipe.dev/models/mediapipe/selfie/v1/'
