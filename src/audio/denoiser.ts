// AudioDenoiser — main-thread controller for the denoise subsystem. Owns the
// AudioContext graph (source → [worklet] → destination) and the worklet
// lifecycle. Independent of the video Worker; the only shared surface is the
// output MediaStream (its destination track).
//
// Lifecycle mirrors the video "passthrough until ready" pattern: the output
// track is live immediately as passthrough; the denoise worklet is spliced in
// once its wasm + weights have loaded.
//
// Fail open: whatever goes wrong (asset fetch, module load, wasm compile or
// instantiate inside the worklet, an exception in process()), the graph goes
// back to source → destination, so the output carries the ORIGINAL audio —
// never silence — and onError says why.
//
// A context that isn't running (autoplay policy, iOS interruption) passes no
// audio at all, so then the output track switches to a clone of the mic track
// (original audio), and back to the context's track once it runs. That is the
// one case where `outputTrack` changes; onOutputTrackChanged says so.

import {
  KERNELS, SAMPLE_RATE, resolveModel,
  type DenoiseModel, type DenoiseModelOption,
} from './kernels'
import { fetchKernel, simdSupported } from './fetch_assets'
import { selectTier } from './probe'
import type { ProcessorInit, FromWorklet, AudioStats } from './messages'
import { WORKLET_SOURCE } from './worklet_inline'
import { DEFAULT_WEIGHTS_BASE_URL } from '~/weights_url.ts'
import { isDebug } from '~/pipeline/debug.ts'

export interface AudioDenoiserOptions {
  model?:          DenoiseModelOption   // 'auto' (default) | tier | explicit model
  weightsBaseUrl?: string               // default: the public CDN (DEFAULT_WEIGHTS_BASE_URL)
  postFilterBeta?: number               // default 0.03 (DFN)
  gruLeak?:        number               // default 0.9995 (DFN): decays GRU state per hop; 1 = off, lower = more suppression, less speech quality
  enabled?:        boolean              // default true
  debug?:          boolean              // default: the Pipeline's `debug` flag
  onReady?:        () => void
  onError?:        (message: string) => void
  // The AudioContext is suspended (autoplay policy) and couldn't be resumed
  // without a user gesture. The output is the original mic audio (see
  // onOutputTrackChanged) until resume() or the next gesture (the denoiser
  // listens for one and resumes by itself).
  onSuspended?:    () => void
  // `outputTrack` was replaced: `denoised` false = the context stopped running
  // and the output is now a clone of the mic track (original audio); true = it
  // runs again and the output is back on the denoiser's track. Put the new
  // track on your sender (RTCRtpSender.replaceTrack).
  onOutputTrackChanged?: (track: MediaStreamTrack, previous: MediaStreamTrack, denoised: boolean) => void
}

const DEFAULT_BETA = 0.03
// 0.995 cost up to 0.7 PESQ on noisy speech. 0.9995 is within ~0.1 of no leak on a
// fresh 20 s clip and ~0.1 better than no leak over a 14-min stream; no drift either way.
const DEFAULT_GRU_LEAK = 0.9995
const GESTURES = ['pointerdown', 'keydown', 'touchend'] as const
// How long the context may stay not-running before the output falls back to
// the original audio (a context created while allowed starts within a few ms).
const SUSPEND_GRACE_MS = 250
// An unlocked context nobody adopts is closed after this long.
const UNLOCK_TTL_MS = 60_000

// A context created (and resumed) inside a user gesture by unlock(), waiting
// for the next AudioDenoiser to adopt it.
let unlocked: { ctx: AudioContext; timer: ReturnType<typeof setTimeout> } | null = null

export class AudioDenoiser {
  // Create (or resume) the AudioContext the next AudioDenoiser will use, NOW.
  // Call it synchronously in the click handler that starts your call, before
  // any await (getUserMedia, fetches): browsers only let an AudioContext start
  // during a user gesture, and the gesture has expired by the time the stream
  // arrives. Safe to call repeatedly or without a gesture (it does nothing
  // useful then). Unused, the context is closed after a minute.
  static unlock(): void {
    if (typeof AudioContext === 'undefined') return
    if (unlocked && unlocked.ctx.state !== 'closed') { void unlocked.ctx.resume().catch(() => {}); return }
    let ctx: AudioContext
    try { ctx = new AudioContext({ sampleRate: SAMPLE_RATE }) } catch {
      try { ctx = new AudioContext() } catch { return }
    }
    void ctx.resume().catch(() => {})
    const entry = {
      ctx,
      timer: setTimeout(() => { if (unlocked === entry) { unlocked = null; void ctx.close().catch(() => {}) } }, UNLOCK_TTL_MS),
    }
    unlocked = entry
  }

  // Resolves true once the denoise worklet is spliced in; false if init failed
  // (or the denoiser was destroyed first) and the graph stayed in passthrough.
  // Never rejects — audio keeps flowing either way; onError carries the reason.
  readonly ready: Promise<boolean>

  private ctx: AudioContext
  private source: MediaStreamAudioSourceNode
  private dest: MediaStreamAudioDestinationNode
  // The context's output (denoised / passthrough) and, while the context isn't
  // running, a clone of the mic track carrying the original audio.
  private readonly destTrack: MediaStreamTrack
  private rawTrack: MediaStreamTrack | null = null
  private current: MediaStreamTrack
  private suspendTimer: ReturnType<typeof setTimeout> | null = null
  private removeStateListener: (() => void) | null = null
  private node: AudioWorkletNode | null = null
  private model: DenoiseModel | null = null
  private latestStats: AudioStats | null = null
  private destroyed = false
  private bypassed = false
  private removeGestureListeners: (() => void) | null = null
  private removeEndedListener: (() => void) | null = null

  // Throws only if no AudioContext graph can be built at all (the caller then
  // passes the original track through).
  constructor(private inputTrack: MediaStreamTrack, private opts: AudioDenoiserOptions = {}) {
    // Request 48 kHz; the browser resamples the mic into the context rate for
    // free when it honors this. If it doesn't, the worklet engages its own
    // resamplers (see worklet/processor.ts). Some engines (historically
    // Firefox) refuse to connect a mic whose rate differs from the context's:
    // retry with the device's default rate before giving up.
    // A context unlock() created inside a gesture is adopted if it got to run;
    // otherwise build one here (the constructor may itself run in a gesture).
    let built = adoptUnlocked(inputTrack)
    if (!built.ok) built = buildGraph(inputTrack, SAMPLE_RATE)
    if (!built.ok) built = buildGraph(inputTrack, undefined)
    if (!built.ok) throw built.error
    this.ctx = built.ctx
    this.source = built.source
    this.dest = this.ctx.createMediaStreamDestination()
    this.destTrack = this.dest.stream.getAudioTracks()[0]
    this.current = this.destTrack

    // Passthrough immediately; denoise splices in when ready.
    this.source.connect(this.dest)

    // Autoplay policy can start the context suspended (Safari, Firefox or
    // Chrome without a recent user gesture) — and a suspended context passes NO
    // audio, not even passthrough. Try to resume now (synchronously: this keeps
    // a gesture the constructor runs in); if the browser refuses, the output
    // falls back to the original mic audio, resumes on the next user gesture,
    // and the caller is told.
    void this.resume().catch(() => { /* handled by the state watcher */ })
    const onState = () => this.onStateChange()
    this.ctx.addEventListener('statechange', onState)
    this.removeStateListener = () => this.ctx.removeEventListener('statechange', onState)
    this.onStateChange()

    // Mic ended (unplugged / stopped) → end our output too.
    const onEnded = () => this.endOutput()
    inputTrack.addEventListener('ended', onEnded)
    this.removeEndedListener = () => inputTrack.removeEventListener('ended', onEnded)

    this.ready = this.init().then(() => this.node !== null && !this.bypassed, (err) => {
      this.opts.onError?.(`audio denoise init failed: ${(err as Error).message ?? String(err)}; passing the original audio through`)
      // Leave passthrough running — audio still flows, just not denoised.
      return false
    })
  }

  // The current output track: the context's (denoised, or passthrough while
  // the worklet loads / after a failure), or the original mic audio while the
  // context isn't running.
  get outputTrack(): MediaStreamTrack { return this.current }

  // Context running → output on the context's track. Not running for
  // SUSPEND_GRACE_MS → original audio on a clone of the mic track, a
  // non-fatal onSuspended, and a resume on the next user gesture.
  private onStateChange(): void {
    if (this.destroyed) return
    const state = this.ctx.state as string
    if (state === 'closed') return
    if (state === 'running') {
      if (this.suspendTimer) { clearTimeout(this.suspendTimer); this.suspendTimer = null }
      this.removeGestureListeners?.()
      this.setOutput(this.destTrack)
      return
    }
    if (this.suspendTimer) return
    this.suspendTimer = setTimeout(() => {
      this.suspendTimer = null
      if (this.destroyed || (this.ctx.state as string) === 'running' || this.ctx.state === 'closed') return
      this.setOutput(this.originalTrack())
      this.opts.onSuspended?.()
      this.listenForGesture()
    }, SUSPEND_GRACE_MS)
  }

  private originalTrack(): MediaStreamTrack {
    if (!this.rawTrack || this.rawTrack.readyState === 'ended') this.rawTrack = this.inputTrack.clone()
    return this.rawTrack
  }

  private setOutput(track: MediaStreamTrack): void {
    if (track === this.current) return
    const previous = this.current
    this.current = track
    try { this.opts.onOutputTrackChanged?.(track, previous, track === this.destTrack) } catch (err) {
      console.error('[longpipe/audio] onOutputTrackChanged threw:', err)
    }
  }

  private listenForGesture(): void {
    if (typeof window === 'undefined' || this.removeGestureListeners) return
    const onGesture = () => { void this.ctx.resume().catch(() => {}) }
    for (const g of GESTURES) window.addEventListener(g, onGesture, { capture: true })
    this.removeGestureListeners = () => {
      for (const g of GESTURES) window.removeEventListener(g, onGesture, { capture: true })
      this.removeGestureListeners = null
    }
  }

  // Explicit model/tier choice (NOT 'auto') → resolve, with a SIMD fallback.
  private resolveExplicit(): DenoiseModel {
    const model = resolveModel(this.opts.model as Exclude<DenoiseModelOption, 'auto'>)
    if (KERNELS[model].needsSimd && !simdSupported()) {
      this.opts.onError?.(`${model} needs wasm SIMD; falling back to rnnoise`)
      return 'rnnoise'
    }
    return model
  }

  private async init(): Promise<void> {
    // 'auto' runs the weight-free tier probe; an explicit model/tier skips it.
    const baseUrl = this.opts.weightsBaseUrl ?? DEFAULT_WEIGHTS_BASE_URL
    const model = (this.opts.model ?? 'auto') === 'auto'
      ? await selectTier(baseUrl, this.opts.onError)
      : this.resolveExplicit()
    this.model = model

    const assets = await fetchKernel(model, baseUrl)
    if (!this.ctx.audioWorklet) throw new Error('AudioWorklet is not available in this context')
    await this.ctx.audioWorklet.addModule(workletUrl())
    if (this.destroyed) return

    const processorOptions: ProcessorInit = {
      model,
      wasmBytes:      assets.wasmBytes,
      weights:        assets.weights,
      enabled:        this.opts.enabled ?? true,
      postFilterBeta: this.opts.postFilterBeta ?? DEFAULT_BETA,
      gruLeak:        this.opts.gruLeak ?? DEFAULT_GRU_LEAK,
      debug:          this.opts.debug ?? isDebug(),
    }
    const node = new AudioWorkletNode(this.ctx, 'denoise', {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], processorOptions,
    })
    node.port.onmessage = (e: MessageEvent<FromWorklet>) => {
      const d = e.data
      if (d.type === 'stats') this.latestStats = d.stats
      else if (d.type === 'error') {
        // fatal = the kernel can't run (wasm compile/instantiate failed): drop
        // the worklet from the graph so nothing depends on it. The processor
        // already passes input through in that state; this removes it.
        if (d.fatal) this.bypass()
        this.opts.onError?.(d.fatal ? `${d.message}; passing the original audio through` : d.message)
      }
      else if (d.type === 'ready') this.opts.onReady?.()
    }
    // An exception inside process() kills the processor: it outputs silence
    // from then on. Route around it.
    node.onprocessorerror = () => {
      this.bypass()
      this.opts.onError?.('audio denoise processor crashed; passing the original audio through')
    }
    this.node = node

    // Splice the worklet in: source → node → dest, dropping the passthrough edge.
    if (this.bypassed) return
    this.source.disconnect(this.dest)
    this.source.connect(node).connect(this.dest)
  }

  // Back to plain source → destination (original audio).
  private bypass(): void {
    if (this.bypassed || this.destroyed) return
    this.bypassed = true
    if (this.node) {
      try { this.source.disconnect(this.node) } catch { /* not connected */ }
      try { this.node.disconnect() } catch { /* not connected */ }
    }
    try { this.source.disconnect(this.dest) } catch { /* not connected */ }
    this.source.connect(this.dest)
  }

  // True when the original audio is being passed through because denoise
  // failed (not because setEnabled(false) asked for it).
  get isBypassed(): boolean { return this.bypassed }

  setEnabled(on: boolean): void {
    this.node?.port.postMessage({ type: 'enabled', value: on })
  }

  setConfig(cfg: { postFilterBeta?: number; gruLeak?: number }): void {
    this.node?.port.postMessage({ type: 'config', ...cfg })
  }

  getStats(): AudioStats | null { return this.latestStats }

  // Context state ('running' | 'suspended' | 'closed').
  get state(): AudioContextState { return this.ctx.state }

  // Resume the internal AudioContext. The constructor already tries this, but
  // autoplay policy may block it until a user gesture — call it from a click
  // handler (e.g. your join / unmute button) to be safe. No-op when already
  // running or after destroy().
  async resume(): Promise<void> {
    if (this.destroyed || this.ctx.state === 'running' || this.ctx.state === 'closed') return
    await this.ctx.resume()
    // statechange moves the output back to the context's track.
  }

  private endOutput(): void {
    if (this.destroyed) return
    this.destroy()
  }

  // Ends the output track too: it's ours (a MediaStreamAudioDestinationNode
  // track), and a closed context leaves it 'live' but silent, so a sender
  // would keep transmitting silence after pipeline.destroy().
  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
    this.destTrack.stop()
    this.rawTrack?.stop()
    if (this.suspendTimer) clearTimeout(this.suspendTimer)
    this.removeStateListener?.()
    this.removeGestureListeners?.()
    this.removeEndedListener?.()
    try { this.node?.disconnect() } catch { /* already gone */ }
    try { this.source.disconnect() } catch { /* already gone */ }
    void this.ctx.close().catch(() => {})
  }
}

type Graph =
  | { ok: true; ctx: AudioContext; source: MediaStreamAudioSourceNode }
  | { ok: false; error: unknown }

// The context unlock() made, if it is running (one that didn't start is no
// better than a fresh one, and on iOS getUserMedia can interrupt it) and the
// mic connects to it. Otherwise it is closed.
function adoptUnlocked(inputTrack: MediaStreamTrack): Graph {
  const entry = unlocked
  unlocked = null
  if (!entry) return { ok: false, error: new Error('no unlocked AudioContext') }
  clearTimeout(entry.timer)
  const ctx = entry.ctx
  if (ctx.state !== 'running') {
    void ctx.close().catch(() => {})
    return { ok: false, error: new Error(`unlocked AudioContext is ${ctx.state}`) }
  }
  try {
    return { ok: true, ctx, source: ctx.createMediaStreamSource(new MediaStream([inputTrack])) }
  } catch (error) {
    void ctx.close().catch(() => {})
    return { ok: false, error }
  }
}

function buildGraph(inputTrack: MediaStreamTrack, sampleRate: number | undefined): Graph {
  let ctx: AudioContext | null = null
  try {
    ctx = sampleRate ? new AudioContext({ sampleRate }) : new AudioContext()
    const source = ctx.createMediaStreamSource(new MediaStream([inputTrack]))
    return { ok: true, ctx, source }
  } catch (error) {
    void ctx?.close().catch(() => {})
    return { ok: false, error }
  }
}

// Blob-URL the bundled processor (published) so addModule is same-origin from
// any CDN consumer; URL fallback in dev (note: AudioWorklet has no ESM import,
// so the dev URL path only works after a build populates WORKLET_SOURCE).
function workletUrl(): string {
  if (WORKLET_SOURCE) {
    return URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }))
  }
  return new URL('./worklet/processor.ts', import.meta.url).href
}
