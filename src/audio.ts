// `longpipe/audio` — the speech denoiser, usable WITHOUT the video pipeline.
//
// For audio-only streams (camera off) or toggling noise suppression from a mic
// settings UI, where only an audio track is in hand and spinning up the GPU
// worker would be wasteful. Documented at https://longpipe.dev/docs/audio.
//
// A SEPARATE ENTRY POINT so it carries none of the video pipeline: no inlined
// worker, no shaders — just the denoiser and its inlined AudioWorklet. This is
// the same AudioDenoiser `EffectsPipeline` uses for `audio: 'denoise'`.
export { AudioDenoiser } from '~/audio/denoiser.ts'
export type { AudioDenoiserOptions } from '~/audio/denoiser.ts'
export type { DenoiseModel, DenoiseTier, DenoiseModelOption } from '~/audio/kernels.ts'
export type { AudioStats } from '~/audio/messages.ts'
