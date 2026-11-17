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

import {
  KERNELS, SAMPLE_RATE, resolveModel,
  type DenoiseModel, type DenoiseModelOption,
} from './kernels'
import { fetchKernel, simdSupported } from './fetch_assets'
import { selectTier } from './probe'
import type { ProcessorInit, FromWorklet, AudioStats } from './messages'
import { WORKLET_SOURCE } from './worklet_inline'
import { DEFAULT_WEIGHTS_BASE_URL } from '~/weights_url.ts'

export interface AudioDenoiserOptions {
  model?:          DenoiseModelOption   // 'auto' (default) | tier | explicit model
  weightsBaseUrl?: string               // default: the public CDN (DEFAULT_WEIGHTS_BASE_URL)
  postFilterBeta?: number               // default 0.03 (DFN)
  gruLeak?:        number               // default 0.995 (DFN)
  enabled?:        boolean              // default true
  onReady?:        () => void
  onError?:        (message: string) => void
  // The AudioContext is suspended (autoplay policy) and couldn't be resumed
  // without a user gesture: output is silent until resume() or the next
  // gesture (the denoiser listens for one and resumes by itself).
  onSuspended?:    () => void
}

const DEFAULT_BETA = 0.03
const DEFAULT_GRU_LEAK = 0.995
const GESTURES = ['pointerdown', 'keydown', 'touchend'] as const

export class AudioDenoiser {
  readonly outputTrack: MediaStreamTrack
  // Resolves true once the denoise worklet is spliced in; false if init failed
  // (or the denoiser was destroyed first) and the graph stayed in passthrough.
  // Never rejects — audio keeps flowing either way; onError carries the reason.
  readonly ready: Promise<boolean>

  private ctx: AudioContext
  private source: MediaStreamAudioSourceNode
  private dest: MediaStreamAudioDestinationNode
  private node: AudioWorkletNode | null = null
  private model: DenoiseModel | null = null
  private latestStats: AudioStats | null = null
  private destroyed = false
  private bypassed = false
  private removeGestureListeners: (() => void) | null = null
  private removeEndedListener: (() => void) | null = null

  // Throws only if no AudioContext graph can be built at all (the caller then
  // passes the original track through).
  constructor(inputTrack: MediaStreamTrack, private opts: AudioDenoiserOptions = {}) {
    // Request 48 kHz; the browser resamples the mic into the context rate for
    // free when it honors this. If it doesn't, the worklet engages its own
    // resamplers (see worklet/processor.ts). Some engines (historically
    // Firefox) refuse to connect a mic whose rate differs from the context's:
    // retry with the device's default rate before giving up.
    let built = buildGraph(inputTrack, SAMPLE_RATE)
    if (!built.ok) built = buildGraph(inputTrack, undefined)
    if (!built.ok) throw built.error
    this.ctx = built.ctx
    this.source = built.source
    this.dest = this.ctx.createMediaStreamDestination()
    this.outputTrack = this.dest.stream.getAudioTracks()[0]

    // Passthrough immediately; denoise splices in when ready.
    this.source.connect(this.dest)

    // Autoplay policy can start the context suspended (Safari, or Chrome before
    // any user interaction) — and a suspended context passes NO audio, not even
    // passthrough. Try to resume now; if the browser refuses, resume on the
    // first user gesture and tell the caller.
    void this.resume().catch(() => { /* handled by the state check below */ })
    queueMicrotask(() => this.watchSuspended())

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

  private watchSuspended(): void {
    if (this.destroyed || this.ctx.state !== 'suspended') return
    // Give the constructor's resume() a moment; still suspended → gesture.
    setTimeout(() => {
      if (this.destroyed || this.ctx.state !== 'suspended') return
      this.opts.onSuspended?.()
      if (typeof window === 'undefined' || this.removeGestureListeners) return
      const onGesture = () => {
        void this.ctx.resume().then(() => {
          if (this.ctx.state === 'running') this.removeGestureListeners?.()
        }).catch(() => {})
      }
      for (const g of GESTURES) window.addEventListener(g, onGesture, { capture: true })
      this.removeGestureListeners = () => {
        for (const g of GESTURES) window.removeEventListener(g, onGesture, { capture: true })
        this.removeGestureListeners = null
      }
    }, 250)
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
    if (this.destroyed || this.ctx.state !== 'suspended') return
    await this.ctx.resume()
    if ((this.ctx.state as AudioContextState) === 'running') this.removeGestureListeners?.()
  }

  private endOutput(): void {
    if (this.destroyed) return
    this.outputTrack.stop()
    this.destroy()
  }

  destroy(): void {
    if (this.destroyed) return
    this.destroyed = true
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
