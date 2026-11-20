// Longpipe effects as LiveKit track processors.
//
//   const processor = BackgroundBlur()                // or VirtualBackground(url), LongpipeProcessor({...})
//   await cameraTrack.setProcessor(processor)
//   await processor.switchTo({ mode: 'virtual-background', imagePath: url })
//
// Each processor wraps one EffectsPipeline (video) or AudioDenoiser (audio).
// LiveKit calls init() with the raw track, restart() when that track is
// replaced (camera switch, unmute), destroy() on stopProcessor/unpublish.
import { EffectsPipeline, type PipelineOptions, type BackgroundInput } from 'longpipe'
import { AudioDenoiser, type AudioDenoiserOptions } from 'longpipe/audio'
import type { Track, TrackProcessor, ProcessorOptions, AudioProcessorOptions, LocalTrack } from 'livekit-client'

/** The modes of `@livekit/track-processors`' `BackgroundProcessor`, so `switchTo` calls carry over. */
export type BackgroundMode =
  | { mode: 'background-blur'; blurRadius?: number }
  | { mode: 'virtual-background'; imagePath: string }
  | { mode: 'disabled' }

/** Any `EffectsPipeline` option except the ones the processor manages (audio, track callbacks). */
export type LongpipeProcessorOptions = Omit<PipelineOptions, 'audio' | 'onTrackChanged' | 'onAudioTrackChanged'>

// LiveKit's blurRadius is a pixel radius (default 10). 10 → sigma 8, Longpipe's medium blur.
const blur = (radius?: number): BackgroundInput => (radius == null ? 'blur' : { blur: { sigma: radius * 0.8 } })

function backgroundFor(m: BackgroundMode): BackgroundInput {
  if (m.mode === 'virtual-background') return m.imagePath
  if (m.mode === 'background-blur') return blur(m.blurRadius)
  return 'none'
}

// Put a track the processor swapped in by itself (not inside init/restart, where
// LiveKit does this) on the senders and local preview elements.
async function swapProcessedTrack(local: LocalTrack | undefined, track: MediaStreamTrack, previous?: MediaStreamTrack) {
  if (!local) return
  await local.sender?.replaceTrack(track).catch(() => {})
  const codecs = (local as { simulcastCodecs?: Map<unknown, { sender?: RTCRtpSender }> }).simulcastCodecs
  for (const sc of codecs?.values() ?? []) await sc.sender?.replaceTrack(track).catch(() => {})
  for (const el of local.attachedElements) {
    const ms = el.srcObject
    if (!(ms instanceof MediaStream)) continue
    if (previous) ms.removeTrack(previous)
    ms.addTrack(track)
  }
}

export class LongpipeVideoProcessor implements TrackProcessor<Track.Kind.Video> {
  name = 'longpipe'
  processedTrack?: MediaStreamTrack
  private _pipeline?: EffectsPipeline
  private local?: LocalTrack

  constructor(private options: LongpipeProcessorOptions = {}) {}

  /** The running EffectsPipeline (after `setProcessor`): `setTouchup`, `setReframe`, `getStats`, `ready`… */
  get pipeline(): EffectsPipeline | undefined { return this._pipeline }

  async init({ track, localTrack }: ProcessorOptions<Track.Kind.Video>): Promise<void> {
    this.local = localTrack
    this.build(track)
  }

  // A new source track (camera switch, unmute). LiveKit stops the old track
  // after this returns and puts `processedTrack` on the sender.
  async restart({ track, localTrack }: ProcessorOptions<Track.Kind.Video>): Promise<void> {
    this.local = localTrack ?? this.local
    const p = this._pipeline
    if (p) {
      // Same output track if the pipeline is still running. A muted camera is
      // stopped by LiveKit, which ends the pipeline: then rebuild.
      const { status } = await p.getStats()
      if (status !== 'ended' && status !== 'destroyed') {
        try { return await p.setInputStream(new MediaStream([track])) } catch { /* rebuild */ }
      }
    }
    this.build(track)
  }

  async destroy(): Promise<void> {
    this._pipeline?.destroy()
    this._pipeline = undefined
    this.processedTrack = undefined
  }

  /** Same call as `BackgroundProcessorWrapper.switchTo` in `@livekit/track-processors`. */
  // Like theirs, it doesn't throw when effects can't run (no GPU, init failed):
  // that already went to onError and the output carries the raw camera.
  async switchTo(mode: BackgroundMode): Promise<void> {
    await this.setBackground(backgroundFor(mode)).catch((e) => {
      if (!(e?.fatal || e?.type === 'no-backend')) throw e
    })
  }

  /** Any Longpipe background: 'blur', 'none', an image URL, `{ color: '#00ff00' }`, … */
  async setBackground(background: BackgroundInput): Promise<void> {
    this.options = { ...this.options, background }
    await this._pipeline?.setBackground(background)
  }

  private build(track: MediaStreamTrack) {
    this._pipeline?.destroy()
    const { onError } = this.options
    // Not awaited: the output carries the raw camera until the effect is live.
    const p: EffectsPipeline = new EffectsPipeline(new MediaStream([track]), {
      ...this.options,
      audio: 'drop',
      // 'camera-ended' also fires when LiveKit stops a muted camera; restart() rebuilds on unmute.
      onError: (e) => { if (e.type !== 'camera-ended') onError?.(e) },
      // Safari only: after a worker crash the output track is replaced.
      onTrackChanged: ({ videoTrack, previousTrack }) => {
        if (p !== this._pipeline) return
        this.processedTrack = videoTrack
        void swapProcessedTrack(this.local, videoTrack, previousTrack)
      },
    })
    this._pipeline = p
    this.processedTrack = p.stream.getVideoTracks()[0]
  }
}

/** Longpipe with any options: background, touch-up, reframe, preset, model, callbacks. */
export const LongpipeProcessor = (options: LongpipeProcessorOptions = {}) => new LongpipeVideoProcessor(options)

/** Drop-in for `BackgroundBlur` from `@livekit/track-processors`. */
export const BackgroundBlur = (blurRadius?: number, options: LongpipeProcessorOptions = {}) =>
  new LongpipeVideoProcessor({ ...options, background: blur(blurRadius) })

/** Drop-in for `VirtualBackground` from `@livekit/track-processors`. */
export const VirtualBackground = (imagePath: string, options: LongpipeProcessorOptions = {}) =>
  new LongpipeVideoProcessor({ ...options, background: imagePath })

/** Drop-in for `BackgroundProcessor({ mode, … })` from `@livekit/track-processors`. */
export const BackgroundProcessor = (opts: BackgroundMode & LongpipeProcessorOptions) => {
  const { mode, blurRadius, imagePath, ...rest } = opts as BackgroundMode & LongpipeProcessorOptions & { blurRadius?: number; imagePath?: string }
  return new LongpipeVideoProcessor({ ...rest, background: backgroundFor(opts) })
}

export type DenoiseProcessorOptions = Omit<AudioDenoiserOptions, 'onOutputTrackChanged'>

/**
 * Speech denoise for a LocalAudioTrack. Uses its own 48 kHz AudioContext (call
 * `EffectsPipeline.unlockAudio()` in the click that starts the call).
 */
export class LongpipeDenoiseProcessor implements TrackProcessor<Track.Kind.Audio, AudioProcessorOptions> {
  name = 'longpipe-denoise'
  processedTrack?: MediaStreamTrack
  private _denoiser?: AudioDenoiser
  private local?: LocalTrack

  constructor(private options: DenoiseProcessorOptions = {}) {}

  get denoiser(): AudioDenoiser | undefined { return this._denoiser }

  async init({ track, localTrack }: AudioProcessorOptions): Promise<void> {
    this.local = localTrack
    this.build(track)
  }

  async restart({ track, localTrack }: AudioProcessorOptions): Promise<void> {
    this.local = localTrack ?? this.local
    this.build(track)
  }

  async destroy(): Promise<void> {
    this._denoiser?.destroy()
    this._denoiser = undefined
    this.processedTrack = undefined
  }

  setEnabled(on: boolean): void {
    this.options = { ...this.options, enabled: on }
    this._denoiser?.setEnabled(on)
  }

  private build(track: MediaStreamTrack) {
    this._denoiser?.destroy()
    const d: AudioDenoiser = new AudioDenoiser(track, {
      ...this.options,
      // The AudioContext isn't running (autoplay policy): the output moved to the
      // original mic audio, and moves back once it runs.
      onOutputTrackChanged: (next, previous) => {
        if (d !== this._denoiser) return
        this.processedTrack = next
        void swapProcessedTrack(this.local, next, previous)
      },
    })
    this._denoiser = d
    this.processedTrack = d.outputTrack
  }
}

export const DenoiseProcessor = (options: DenoiseProcessorOptions = {}) => new LongpipeDenoiseProcessor(options)

export { EffectsPipeline }
