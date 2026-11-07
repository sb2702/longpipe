// AudioDenoiser — main-thread controller for the denoise subsystem. Owns the
// AudioContext graph (source → [worklet] → destination) and the worklet
// lifecycle. Independent of the video Worker; the only shared surface is the
// output MediaStream (its destination track).
//
// Lifecycle mirrors the video "passthrough until ready" pattern: the output
// track is live immediately as passthrough; the denoise worklet is spliced in
// once its wasm + weights have loaded.

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
}

const DEFAULT_BETA = 0.03
const DEFAULT_GRU_LEAK = 0.995

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

  constructor(inputTrack: MediaStreamTrack, private opts: AudioDenoiserOptions = {}) {
    // Request 48 kHz; the browser resamples the mic into the context rate for
    // free when it honors this. If it doesn't, the worklet engages its own
    // resamplers (see worklet/processor.ts).
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    this.source = this.ctx.createMediaStreamSource(new MediaStream([inputTrack]))
    this.dest = this.ctx.createMediaStreamDestination()
    this.outputTrack = this.dest.stream.getAudioTracks()[0]

    // Autoplay policy can start the context suspended (Safari, or Chrome before
    // any user interaction) — and a suspended context passes NO audio, not even
    // passthrough. Try to resume now; if the browser refuses, the caller can
    // retry from a user gesture via resume().
    void this.resume().catch(() => { /* blocked until a gesture; see resume() */ })

    // Passthrough immediately; denoise splices in when ready.
    this.source.connect(this.dest)
    this.ready = this.init().then(() => this.node !== null, (err) => {
      this.opts.onError?.(`audio denoise init failed: ${(err as Error).message ?? String(err)}`)
      // Leave passthrough running — audio still flows, just not denoised.
      return false
    })
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
      else if (d.type === 'error') this.opts.onError?.(d.message)
      else if (d.type === 'ready') this.opts.onReady?.()
    }
    this.node = node

    // Splice the worklet in: source → node → dest, dropping the passthrough edge.
    this.source.disconnect(this.dest)
    this.source.connect(node).connect(this.dest)
  }

  setEnabled(on: boolean): void {
    this.node?.port.postMessage({ type: 'enabled', value: on })
  }

  setConfig(cfg: { postFilterBeta?: number; gruLeak?: number }): void {
    this.node?.port.postMessage({ type: 'config', ...cfg })
  }

  getStats(): AudioStats | null { return this.latestStats }

  // Resume the internal AudioContext. The constructor already tries this, but
  // autoplay policy may block it until a user gesture — call it from a click
  // handler (e.g. your join / unmute button) to be safe. No-op when already
  // running or after destroy().
  async resume(): Promise<void> {
    if (this.destroyed || this.ctx.state !== 'suspended') return
    await this.ctx.resume()
  }

  destroy(): void {
    this.destroyed = true
    try { this.node?.disconnect() } catch { /* already gone */ }
    try { this.source.disconnect() } catch { /* already gone */ }
    void this.ctx.close()
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
