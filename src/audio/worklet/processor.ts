// Denoise AudioWorkletProcessor — runs on the audio render thread.
//
// The bare wasm Module is compiled on the main thread and handed in via
// processorOptions; we instantiate it here (the worklet scope has no fetch /
// no ESM). process() delivers 128-sample render quanta but the kernels work on
// 480-sample (10 ms @ 48 kHz) hops, so an input accumulator + a primed output
// ring decouple the two.
//
// When the AudioContext can't run at 48 kHz, two stateful rubato resamplers
// (exposed by the DFN wasm) bridge device-rate ⇄ 48 kHz around the net. They
// engage only when sampleRate ≠ 48000 and the kernel exposes them (DFN does;
// rnnoise doesn't — at non-48k on rnnoise we error rather than corrupt).
//
// Bundled to a self-contained ESM by the inline-worklet tsup plugin and loaded
// via Blob URL + audioWorklet.addModule (see ../worklet_inline.ts).

import { KERNELS, HOP, SAMPLE_RATE, type KernelSpec } from '~/audio/kernels.ts'
import type { ProcessorInit, ToWorklet, FromWorklet, AudioStats } from '~/audio/messages.ts'

declare const sampleRate: number
declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort
  constructor()
}
declare function registerProcessor(
  name: string,
  ctor: new (options: { processorOptions: ProcessorInit }) => AudioWorkletProcessor,
): void

type Fn = (...a: number[]) => number
type WasmExports = Record<string, Fn> & { memory: WebAssembly.Memory }
const RS_CAP = 4096   // resampler output-burst scratch capacity (frames)

class DenoiseProcessor extends AudioWorkletProcessor {
  private cfg: KernelSpec
  private mem!: WebAssembly.Memory
  private run!: (st: number, inPtr: number, outPtr: number) => number
  private state = 0
  private inPtr = 0
  private outPtr = 0
  private setBeta: Fn | null = null
  private setGruLeak: Fn | null = null

  // Resampling (engaged only when sampleRate ≠ 48000 and the wasm supports it).
  private resample = false
  private rsPush: ((rs: number, inPtr: number, n: number, outPtr: number, cap: number) => number) | null = null
  private inRs = 0
  private outRs = 0
  private rsDevInPtr = 0   // device-rate input scratch (one quantum)
  private rsOutPtr = 0     // resampler output-burst scratch (RS_CAP)

  private enabled: boolean
  private model: ProcessorInit['model']

  // 128 ⇄ 480 buffering (at 48 kHz). outRing is at the OUTPUT (device) rate.
  private inAcc = new Float32Array(HOP)
  private inFill = 0
  private outRing = new Float32Array(4096)
  private outRead = 0
  private outWrite = 0
  private outCount = 0
  private primed = false

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

    try {
      const ex = this.instantiate(init.wasmBytes)
      const pick = (name?: string) => (name ? ex[name] : undefined)
      this.mem = ex.memory
      const malloc = ex[this.cfg.exports.malloc]
      this.run = ex[this.cfg.exports.process] as typeof this.run
      this.inPtr = malloc(HOP * 4)
      this.outPtr = malloc(HOP * 4)

      if (init.weights) this.state = ex[this.cfg.exports.create](this.upload(malloc, init.weights), init.weights.byteLength)
      else this.state = ex[this.cfg.exports.create]()

      this.setBeta = (pick(this.cfg.exports.setBeta) ?? null) as Fn | null
      this.setGruLeak = (pick(this.cfg.exports.setGruLeak) ?? null) as Fn | null
      if (this.setBeta && (init.postFilterBeta ?? 0) > 0) this.setBeta(this.state, init.postFilterBeta!)
      if (this.setGruLeak && (init.gruLeak ?? 1) < 1) this.setGruLeak(this.state, init.gruLeak!)

      // Non-48k → wire the resamplers if this wasm provides them.
      if (sampleRate !== SAMPLE_RATE) {
        const rsCreate = ex['df_resampler_create']
        this.rsPush = (ex['df_resampler_push'] as typeof this.rsPush) ?? null
        if (rsCreate && this.rsPush) {
          this.resample = true
          this.inRs = rsCreate(sampleRate, SAMPLE_RATE)      // device → 48k
          this.outRs = rsCreate(SAMPLE_RATE, sampleRate)     // 48k → device
          this.rsDevInPtr = malloc(256 * 4)
          this.rsOutPtr = malloc(RS_CAP * 4)
        } else {
          this.post({ type: 'error', message: `${init.model} can't resample ${sampleRate}→48000 Hz; needs a 48 kHz AudioContext` })
        }
      }
    } catch (err) {
      const e = err as Error
      this.post({ type: 'error', message: `denoise init failed: ${e.message ?? String(err)}${e.stack ? '\n' + e.stack : ''}` })
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

  private instantiate(bytes: ArrayBuffer): WasmExports {
    const module = new WebAssembly.Module(bytes)   // sync compile — fine off the main thread
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
    new Uint8Array(this.mem.buffer).set(u8, ptr)
    return ptr
  }

  private heap(): Float32Array {
    if (!this._f32 || this._f32.buffer !== this.mem.buffer) this._f32 = new Float32Array(this.mem.buffer)
    return this._f32
  }

  private post(msg: FromWorklet): void { this.port.postMessage(msg) }

  private pushOut(v: number): void {
    this.outRing[this.outWrite] = v
    this.outWrite = (this.outWrite + 1) % this.outRing.length
    this.outCount++
  }

  // Accumulate 48 kHz samples; run a hop each time 480 fill.
  private feed48k(s: ArrayLike<number>, n: number): void {
    for (let i = 0; i < n; i++) {
      this.inAcc[this.inFill++] = s[i]
      if (this.inFill === HOP) { this.runHop(); this.inFill = 0 }
    }
  }

  private runHop(): void {
    let f32 = this.heap()
    const inBase = this.inPtr >> 2
    for (let i = 0; i < HOP; i++) f32[inBase + i] = this.inAcc[i] * this.cfg.scaleIn

    const t0 = this.now ? this.now() : 0
    this.run(this.state, this.inPtr, this.outPtr)
    if (this.now) this.times[this.tIdx = (this.tIdx + 1) & 255] = this.now() - t0

    if (this.resample && this.rsPush) {
      // outPtr holds the 48 kHz net output (DFN scaleOut === 1) — resample it to
      // device rate and ring the result. df_resampler_push may grow memory.
      const got = this.rsPush(this.outRs, this.outPtr, HOP, this.rsOutPtr, RS_CAP)
      f32 = this.heap()
      const ob = this.rsOutPtr >> 2
      for (let i = 0; i < got; i++) this.pushOut(f32[ob + i])
    } else {
      const ob = this.outPtr >> 2
      for (let i = 0; i < HOP; i++) this.pushOut(f32[ob + i] * this.cfg.scaleOut)
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
      latencyMs: (this.outCount / sampleRate) * 1000,
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

    // Feed the 480-hop accumulator with 48 kHz samples — resampled from the
    // device rate when needed, else the input directly.
    if (this.resample && this.rsPush) {
      let f32 = this.heap()
      const ib = this.rsDevInPtr >> 2
      for (let i = 0; i < n; i++) f32[ib + i] = inCh[i]
      const got = this.rsPush(this.inRs, this.rsDevInPtr, n, this.rsOutPtr, RS_CAP)
      f32 = this.heap()
      this.feed48k(f32.subarray(this.rsOutPtr >> 2, (this.rsOutPtr >> 2) + got), got)
    } else {
      this.feed48k(inCh, n)
    }

    // Prime one hop before draining so the device-rate ring never underruns.
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
