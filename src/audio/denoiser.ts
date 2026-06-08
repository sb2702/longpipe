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
import type { ProcessorInit, FromWorklet, AudioStats } from './messages'
import { WORKLET_SOURCE } from './worklet_inline'

export interface AudioDenoiserOptions {
  model:           DenoiseModelOption   // 'auto' | tier | explicit model
  weightsBaseUrl:  string
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
  readonly ready: Promise<void>

  private ctx: AudioContext
  private source: MediaStreamAudioSourceNode
  private dest: MediaStreamAudioDestinationNode
  private node: AudioWorkletNode | null = null
  private model: DenoiseModel | null = null
  private latestStats: AudioStats | null = null
  private destroyed = false

  constructor(inputTrack: MediaStreamTrack, private opts: AudioDenoiserOptions) {
    // Request 48 kHz; the browser resamples the mic into the context rate for
    // free when it honors this. (Non-48k fallback → resampler, next increment.)
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    this.source = this.ctx.createMediaStreamSource(new MediaStream([inputTrack]))
    this.dest = this.ctx.createMediaStreamDestination()
    this.outputTrack = this.dest.stream.getAudioTracks()[0]

    // Passthrough immediately; denoise splices in when ready.
    this.source.connect(this.dest)
    this.ready = this.init().catch((err) => {
      this.opts.onError?.(`audio denoise init failed: ${(err as Error).message ?? String(err)}`)
      // Leave passthrough running — audio still flows, just not denoised.
    })
  }

  private resolveModelChoice(): DenoiseModel {
    // 'auto' is a capability gate for now (DFN high if SIMD, else RNNoise floor).
    // The weight-free calibration probe (probe.ts) replaces this before publish.
    if (this.opts.model === 'auto') return simdSupported() ? 'dfn' : 'rnnoise'
    const model = resolveModel(this.opts.model)
    // Explicit DFN on a no-SIMD device can't run — fall back with a warning.
    if (KERNELS[model].needsSimd && !simdSupported()) {
      this.opts.onError?.(`${model} needs wasm SIMD; falling back to rnnoise`)
      return 'rnnoise'
    }
    return model
  }

  private async init(): Promise<void> {
    const model = this.resolveModelChoice()
    this.model = model

    const assets = await fetchKernel(model, this.opts.weightsBaseUrl)
    await this.ctx.audioWorklet.addModule(workletUrl())
    if (this.destroyed) return

    const processorOptions: ProcessorInit = {
      model,
      module:         assets.module,
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
