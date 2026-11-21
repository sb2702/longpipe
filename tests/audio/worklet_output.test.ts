import { describe, it, expect } from 'vitest'
import { fetchKernel } from '~/audio/fetch_assets'
import type { DenoiseModel } from '~/audio/kernels'
import type { ProcessorInit } from '~/audio/messages'

// Renders a clip through the real denoise AudioWorklet processor in an
// OfflineAudioContext and checks every output sample. Regression test for the
// 0.0.22 bug where DFN's first hop came out as 480 NaN samples (a Float32Array
// view went stale when the kernel grew wasm memory).
//
// Needs the pinned kernels: `npm run e2e:assets` (tests/e2e/fetch_assets.mjs).
const ASSETS = '/examples/vanilla-webrtc/public/assets/models/'
const SR = 48000

// ~1 s of speech-like signal: a vibrato harmonic stack under a syllable
// envelope, plus white noise. Seeded, so runs are identical.
function noisySpeech(seconds = 1): Float32Array {
  const x = new Float32Array(Math.round(SR * seconds))
  let seed = 1
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1
  let ph = 0
  for (let i = 0; i < x.length; i++) {
    const t = i / SR
    ph += 2 * Math.PI * (140 + 20 * Math.sin(2 * Math.PI * 3 * t)) / SR
    const env = Math.max(0, Math.sin(2 * Math.PI * 2.5 * t))
    let v = 0
    for (let h = 1; h <= 8; h++) v += Math.sin(h * ph) / h
    x[i] = 0.15 * env * v + 0.03 * rnd()
  }
  return x
}

async function render(model: DenoiseModel, input: Float32Array): Promise<{ out: Float32Array; errors: string[] }> {
  let assets
  try { assets = await fetchKernel(model, ASSETS) } catch (e) {
    throw new Error(`${(e as Error).message} — run \`npm run e2e:assets\` first`)
  }
  const ctx = new OfflineAudioContext(1, input.length, SR)
  await ctx.audioWorklet.addModule(new URL('../../src/audio/worklet/processor.ts', import.meta.url).href)
  const buf = ctx.createBuffer(1, input.length, SR)
  buf.copyToChannel(input, 0)
  const src = ctx.createBufferSource()
  src.buffer = buf
  const processorOptions: ProcessorInit = {
    model, wasmBytes: assets.wasmBytes, weights: assets.weights, enabled: true,
    postFilterBeta: 0.03, gruLeak: 1,
  }
  const node = new AudioWorkletNode(ctx, 'denoise', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], processorOptions,
  })
  const errors: string[] = []
  node.port.onmessage = (e) => { if (e.data?.type === 'error') errors.push(e.data.message) }
  src.connect(node).connect(ctx.destination)
  src.start()
  const rendered = await ctx.startRendering()
  return { out: rendered.getChannelData(0), errors }
}

function firstNonFinite(y: Float32Array): number {
  for (let i = 0; i < y.length; i++) if (!Number.isFinite(y[i])) return i
  return -1
}

function rms(y: Float32Array): number {
  let s = 0
  for (const v of y) s += v * v
  return Math.sqrt(s / Math.max(1, y.length))
}

describe('denoise worklet output', () => {
  for (const model of ['dfnint8', 'rnnoise'] as const) {
    it(`${model}: every sample is finite, from the first one on`, async () => {
      const { out, errors } = await render(model, noisySpeech())
      expect(errors).toEqual([])
      expect(firstNonFinite(out)).toBe(-1)
      // And it actually processed audio (not silence end to end).
      expect(rms(out.subarray(SR / 2))).toBeGreaterThan(1e-3)
    })

    it(`${model}: non-finite input never reaches the output, and denoise recovers`, async () => {
      const x = noisySpeech()
      x.fill(NaN, 12000, 12010)
      x[12020] = Infinity
      x[12021] = -Infinity
      const { out } = await render(model, x)
      expect(firstNonFinite(out)).toBe(-1)
      expect(rms(out.subarray(Math.round(SR * 0.6)))).toBeGreaterThan(1e-3)
    })
  }
})
