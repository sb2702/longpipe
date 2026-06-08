// Denoise AudioWorkletProcessor — runs on the audio render thread.
//
// The bare wasm Module is compiled on the main thread and handed in via
// processorOptions; we instantiate it here (the worklet scope has no fetch /
// no ESM). process() delivers 128-sample render quanta but the kernels work on
// 480-sample (10 ms @ 48 kHz) hops and 480/128 = 3.75, so an input accumulator
// + a primed output ring decouple the two. See the audio-denoising worklet PoC.
//
// Bundled to a self-contained ESM by the inline-worklet tsup plugin and loaded
// via Blob URL + audioWorklet.addModule (see ../worklet_inline.ts).

import { KERNELS, HOP, SAMPLE_RATE, type KernelSpec } from '~/audio/kernels.ts'
import type { ProcessorInit, ToWorklet, AudioStats } from '~/audio/messages.ts'

// AudioWorklet globals — not in lib.dom. Minimal ambient declarations avoid an
// @types/audioworklet dependency.
declare const sampleRate: number
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort
  constructor()
}
declare function registerProcessor(
  name: string,
  ctor: new (options: { processorOptions: ProcessorInit }) => AudioWorkletProcessor,
): void

type WasmExports = Record<string, (...a: number[]) => number> & { memory: WebAssembly.Memory }

class DenoiseProcessor extends AudioWorkletProcessor {
  private cfg: KernelSpec
  private mem!: WebAssembly.Memory
  private run!: (st: number, inPtr: number, outPtr: number) => number
  private state = 0
  private inPtr = 0
  private outPtr = 0
  private setBeta: ((st: number, v: number) => number) | null = null
  private setGruLeak: ((st: number, v: number) => number) | null = null

  private enabled: boolean
  private model: ProcessorInit['model']

  // 128 ⇄ 480 buffering.
  private inAcc = new Float32Array(HOP)
  private inFill = 0
  private outRing = new Float32Array(2048)
  private outRead = 0
  private outWrite = 0
  private outCount = 0
  private primed = false

  // p50/p95 telemetry.
  private now: (() => number) | null
  private times = new Float32Array(256)
  private tIdx = 0
  private hops = 0
  private _f32: Float32Array | null = null

  constructor(options: { processorOptions: ProcessorInit }) {
    super()
    const init = options.processorOptions
    this.model = init.model
    this.enabled = init.enabled
    this.cfg = KERNELS[init.model]

    // 48 kHz only for now; the streaming resampler is the next increment. Fail
    // loud rather than silently corrupt the net with wrong-rate samples.
    if (sampleRate !== SAMPLE_RATE) {
      this.post({ type: 'error', message: `AudioContext is ${sampleRate} Hz; denoise needs ${SAMPLE_RATE} Hz (resampler not yet wired)` })
    }

    try {
      const ex = this.instantiate(init.module)
      const pick = (name?: string) => (name ? ex[name] : undefined)
      this.mem = ex.memory
      const malloc = ex[this.cfg.exports.malloc]
      this.run = ex[this.cfg.exports.process] as typeof this.run
      this.inPtr = malloc(HOP * 4)
      this.outPtr = malloc(HOP * 4)

      // Build streaming state. DFN uploads the weights pack; rnnoise is parameterless.
      if (init.weights) {
        const wptr = this.upload(malloc, init.weights)
        this.state = ex[this.cfg.exports.create](wptr, init.weights.byteLength)
      } else {
        this.state = ex[this.cfg.exports.create]()
      }

      this.setBeta = (pick(this.cfg.exports.setBeta) ?? null) as typeof this.setBeta
      this.setGruLeak = (pick(this.cfg.exports.setGruLeak) ?? null) as typeof this.setGruLeak
      // Apply initial settings before the first hop (GRU-leak active from frame 1
      // *prevents* drift rather than correcting it later).
      if (this.setBeta && (init.postFilterBeta ?? 0) > 0) this.setBeta(this.state, init.postFilterBeta!)
      if (this.setGruLeak && (init.gruLeak ?? 1) < 1) this.setGruLeak(this.state, init.gruLeak!)
    } catch (err) {
      this.post({ type: 'error', message: `denoise init failed: ${(err as Error).message ?? String(err)}` })
    }

    this.now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : null

    this.port.onmessage = (e: MessageEvent<ToWorklet>) => {
      const d = e.data
      if (d.type === 'enabled') this.enabled = d.value
      else if (d.type === 'config') {
        if (this.setBeta && d.postFilterBeta != null) this.setBeta(this.state, d.postFilterBeta)
        if (this.setGruLeak && d.gruLeak != null) this.setGruLeak(this.state, d.gruLeak)
      }
    }
    this.post({ type: 'ready' })
  }

  // Instantiate the bare wasm, auto-stubbing any imports (a standalone module
  // references at most a couple of wasi/clock fns the denoise path never hits).
  private instantiate(module: WebAssembly.Module): WasmExports {
    const imports: Record<string, Record<string, unknown>> = {}
    for (const im of WebAssembly.Module.imports(module)) {
      ;(imports[im.module] ??= {})
      if (im.kind === 'function') imports[im.module][im.name] = () => 0
      else if (im.kind === 'memory') imports[im.module][im.name] = new WebAssembly.Memory({ initial: 256 })
      else if (im.kind === 'table') imports[im.module][im.name] = new WebAssembly.Table({ initial: 0, element: 'anyfunc' })
      else if (im.kind === 'global') imports[im.module][im.name] = new WebAssembly.Global({ value: 'i32', mutable: false }, 0)
    }
    return new WebAssembly.Instance(module, imports as WebAssembly.Imports).exports as WasmExports
  }

  private upload(malloc: (n: number) => number, buf: ArrayBuffer): number {
    const u8 = new Uint8Array(buf)
    const ptr = malloc(u8.length)
    new Uint8Array(this.mem.buffer).set(u8, ptr)   // re-view after malloc (may have grown)
    return ptr
  }

  private heap(): Float32Array {
    if (!this._f32 || this._f32.buffer !== this.mem.buffer) this._f32 = new Float32Array(this.mem.buffer)
    return this._f32
  }

  private post(msg: import('~/audio/messages.ts').FromWorklet): void { this.port.postMessage(msg) }

  private runHop(): void {
    const f32 = this.heap()
    const inBase = this.inPtr >> 2
    for (let i = 0; i < HOP; i++) f32[inBase + i] = this.inAcc[i] * this.cfg.scaleIn

    const t0 = this.now ? this.now() : 0
    this.run(this.state, this.inPtr, this.outPtr)
    if (this.now) this.times[this.tIdx = (this.tIdx + 1) & 255] = this.now() - t0

    const outBase = this.outPtr >> 2
    for (let i = 0; i < HOP; i++) {
      this.outRing[this.outWrite] = f32[outBase + i] * this.cfg.scaleOut
      this.outWrite = (this.outWrite + 1) % this.outRing.length
      this.outCount++
    }
    if (++this.hops >= 50) { this.reportStats(); this.hops = 0 }
  }

  private reportStats(): void {
    const v = Array.from(this.times).filter((x) => x > 0).sort((a, b) => a - b)
    const q = (p: number) => (v.length ? v[Math.min(v.length - 1, Math.floor(p * v.length))] : null)
    const stats: AudioStats = {
      model: this.model,
      p50Ms: this.now ? q(0.5) : null,
      p95Ms: this.now ? q(0.95) : null,
      latencyMs: (this.outCount / SAMPLE_RATE) * 1000,
      active: this.enabled && this.state !== 0,
      sampleRate,
    }
    this.post({ type: 'stats', stats })
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const inCh = inputs[0]?.[0]
    const outCh = outputs[0]?.[0]
    if (!outCh) return true
    const n = outCh.length

    if (!inCh || !this.enabled || this.state === 0) {
      if (inCh && !this.enabled) outCh.set(inCh)   // dry passthrough
      else outCh.fill(0)
      return true
    }

    for (let i = 0; i < n; i++) {
      this.inAcc[this.inFill++] = inCh[i]
      if (this.inFill === HOP) { this.runHop(); this.inFill = 0 }
    }

    // Prime one full hop before draining so the ring never underruns in steady
    // state (production is bursty: 480 every ~3.75 quanta, drained 128/quantum).
    if (!this.primed) {
      if (this.outCount >= HOP) this.primed = true
      else { outCh.fill(0); return true }
    }
    if (this.outCount >= n) {
      for (let i = 0; i < n; i++) {
        outCh[i] = this.outRing[this.outRead]
        this.outRead = (this.outRead + 1) % this.outRing.length
        this.outCount--
      }
    } else {
      outCh.fill(0)
    }
    return true
  }
}

registerProcessor('denoise', DenoiseProcessor)
