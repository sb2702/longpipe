import { describe, it, expect, afterEach } from 'vitest'
import { userEvent } from '@vitest/browser/context'
import { AudioDenoiser } from '~/audio/denoiser'
import { Pipeline } from '~/pipeline/index'
import { syntheticCamera, stubWeights, restoreFetch, errorLog, SMALL, sleep, waitUntil } from '../helpers/pipeline'

// A 440 Hz tone as a mic track.
function toneTrack() {
  const ctx = new AudioContext()
  const osc = ctx.createOscillator()
  osc.frequency.value = 440
  const dest = ctx.createMediaStreamDestination()
  osc.connect(dest)
  osc.start()
  return { track: dest.stream.getAudioTracks()[0], stop: () => { osc.stop(); void ctx.close() } }
}

// RMS level of a track over ~ms.
async function level(track: MediaStreamTrack, ms = 400): Promise<number> {
  const ctx = new AudioContext()
  await ctx.resume()
  const src = ctx.createMediaStreamSource(new MediaStream([track]))
  const an = ctx.createAnalyser()
  an.fftSize = 2048
  src.connect(an)
  const buf = new Float32Array(an.fftSize)
  let peak = 0
  const end = performance.now() + ms
  while (performance.now() < end) {
    await sleep(50)
    an.getFloatTimeDomainData(buf)
    let s = 0
    for (const v of buf) s += v * v
    peak = Math.max(peak, Math.sqrt(s / buf.length))
  }
  void ctx.close()
  return peak
}

let cleanup: (() => void)[] = []
afterEach(() => { for (const f of cleanup.reverse()) { try { f() } catch {} } cleanup = []; restoreFetch() })

// Headless Chromium enforces the autoplay policy: AudioContexts start
// suspended until a user gesture. The first test exercises exactly that path;
// its click also unlocks audio for the rest of the file.
describe('suspended AudioContext (autoplay policy)', () => {
  it('reports audio-suspended, then resumes by itself on the first user gesture', async () => {
    await stubWeights({ 'rnnoise.wasm': () => new Response(null, { status: 404 }) })
    const cam = syntheticCamera(30)
    const tone = toneTrack()
    const log = errorLog()
    const p = new Pipeline(new MediaStream([cam.track, tone.track]), { preset: SMALL, audio: { denoise: { model: 'rnnoise' } }, onError: log.onError })
    cleanup.push(() => p.destroy(), cam.stop, tone.stop)
    const d = (p as unknown as { denoiser: AudioDenoiser & { ctx: AudioContext } }).denoiser
    const e = await log.waitFor('audio-suspended', 3000)
    expect(e.fatal).toBe(false)
    expect(d.state).toBe('suspended')
    // The denoiser listens for the next gesture and resumes on it. (Whether
    // the browser then grants the resume depends on user activation, which a
    // parallel test iframe doesn't reliably get — so assert the call.)
    let resumed = 0
    const real = d.ctx.resume.bind(d.ctx)
    d.ctx.resume = () => { resumed++; return real() }
    window.dispatchEvent(new PointerEvent('pointerdown'))
    expect(resumed).toBe(1)
    // Unlock audio for the rest of this file.
    try { await userEvent.click(document.body) } catch { /* best effort */ }
  })
})

describe('audio denoise fails open (original audio, never silence)', () => {
  it('corrupt wasm (fetches OK, fails to instantiate in the worklet)', async () => {
    await stubWeights({ 'rnnoise.wasm': () => new Response(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 99, 99])) })
    const tone = toneTrack()
    const errors: string[] = []
    const d = new AudioDenoiser(tone.track, { model: 'rnnoise', onError: (m) => errors.push(m) })
    cleanup.push(() => d.destroy(), tone.stop)
    await d.ready
    await waitUntil(() => errors.some(m => m.includes('passing the original audio through')), 5000, 'fatal worklet error')
    expect(d.isBypassed).toBe(true)
    expect(await level(d.outputTrack)).toBeGreaterThan(0.05)
  })

  it('wasm 404: stays in passthrough', async () => {
    await stubWeights({ 'rnnoise.wasm': () => new Response(null, { status: 404 }) })
    const tone = toneTrack()
    const errors: string[] = []
    const d = new AudioDenoiser(tone.track, { model: 'rnnoise', onError: (m) => errors.push(m) })
    cleanup.push(() => d.destroy(), tone.stop)
    expect(await d.ready).toBe(false)
    expect(errors.join()).toContain('init failed')
    expect(await level(d.outputTrack)).toBeGreaterThan(0.05)
  })

  it('AudioContext constructor throws: pipeline passes the original audio track through', async () => {
    await stubWeights()
    const cam = syntheticCamera(30)
    const tone = toneTrack()
    const input = new MediaStream([cam.track, tone.track])
    const Real = globalThis.AudioContext
    ;(globalThis as { AudioContext: unknown }).AudioContext = class { constructor() { throw new DOMException('rate mismatch', 'NotSupportedError') } }
    const log = errorLog()
    let p: Pipeline
    try {
      p = new Pipeline(input, { preset: SMALL, audio: 'denoise', onError: log.onError })
    } finally {
      ;(globalThis as { AudioContext: unknown }).AudioContext = Real
    }
    cleanup.push(() => p.destroy(), cam.stop, tone.stop)
    expect(p.stream.getAudioTracks()[0]).toBe(tone.track)
    const e = await log.waitFor('audio')
    expect(e.fatal).toBe(false)
  })

  it('resumeAudio() resumes a suspended context', async () => {
    await stubWeights({ 'rnnoise.wasm': () => new Response(null, { status: 404 }) })
    const cam = syntheticCamera(30)
    const tone = toneTrack()
    const p = new Pipeline(new MediaStream([cam.track, tone.track]), { preset: SMALL, audio: { denoise: { model: 'rnnoise' } } })
    cleanup.push(() => p.destroy(), cam.stop, tone.stop)
    const d = (p as unknown as { denoiser: AudioDenoiser & { ctx: AudioContext } }).denoiser
    await d.ctx.suspend()
    expect(d.state).toBe('suspended')
    await p.resumeAudio()
    expect(d.state).toBe('running')
  })
})
