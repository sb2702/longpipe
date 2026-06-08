// Audio handling — main thread only. The video Worker never sees audio.
//
// 'passthrough': output carries the input's audio tracks unchanged.
// 'drop':        output is video-only.
// 'denoise':     input audio runs through the AudioDenoiser (a separate
//                AudioWorklet subsystem); its denoised track is the output's
//                audio. See src/audio/ — independent of the video worker.

import type { DenoiseModelOption } from '~/audio/kernels.ts'

export interface DenoiseOptions {
  // 'auto' (probe picks), a tier ('high'|'mid'|'low'), or an explicit model
  // ('rnnoise'|'dfn'|'dfnint8'). Default 'auto'.
  model?:          DenoiseModelOption
  postFilterBeta?: number   // DFN post-filter (residual suppression). Default 0.03.
  gruLeak?:        number   // DFN GRU-leak (bounds recurrent drift). Default 0.995.
  enabled?:        boolean  // start denoising vs. passthrough. Default true.
}

// The public `audio` option. String shorthands + a rich denoise object, mirroring
// how `background` accepts a keyword or a structured config.
export type AudioInput = 'passthrough' | 'drop' | 'denoise' | { denoise: DenoiseOptions }

// Back-compat alias (was the whole type pre-denoise).
export type AudioMode = 'passthrough' | 'drop'

export interface NormalizedAudio {
  mode:     'passthrough' | 'drop' | 'denoise'
  denoise?: DenoiseOptions
}

export function normalizeAudio(input: AudioInput): NormalizedAudio {
  if (input === 'denoise') return { mode: 'denoise', denoise: {} }
  if (typeof input === 'object') return { mode: 'denoise', denoise: input.denoise }
  return { mode: input }
}

// Assemble the output MediaStream. For 'denoise' the caller passes the
// AudioDenoiser's output track (which is live immediately as passthrough and
// swaps to denoised when the worklet is ready).
export function buildOutputStream(
  videoTrack:     MediaStreamTrack,
  inputStream:    MediaStream,
  audio:          NormalizedAudio,
  denoisedTrack?: MediaStreamTrack,
): MediaStream {
  const tracks: MediaStreamTrack[] = [videoTrack]
  if (audio.mode === 'passthrough') tracks.push(...inputStream.getAudioTracks())
  else if (audio.mode === 'denoise' && denoisedTrack) tracks.push(denoisedTrack)
  return new MediaStream(tracks)
}
