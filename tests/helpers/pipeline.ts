// Helpers for end-to-end EffectsPipeline tests (vitest browser mode, Chromium).
//
//   syntheticCamera(fps)  — an animated canvas captured as a MediaStream
//   countFrames(track, ms) — frames that arrive on a track during a window
//   stubWeights()          — serves tests/fixtures/tier_small.bin for every
//                            model_*.bin request (no network)
//   SMALL                  — the fixed preset that matches that fixture

import binUrl from '../fixtures/tier_small.bin?url'
import { PRESETS, type ManualPreset } from '~/pipeline/presets'
import { resetTuning } from '~/pipeline/tuning'
import type { PipelineError } from '~/pipeline/errors'

export const SMALL: ManualPreset = PRESETS[1]

export const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

export function syntheticCamera(fps = 30, w = 640, h = 360) {
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')!
  let n = 0
  const draw = () => {
    ctx.fillStyle = `hsl(${(n++ * 9) % 360}, 70%, 50%)`
    ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = '#fff'
    ctx.fillRect((n * 7) % w, h / 3, 80, 80)
  }
  draw()
  const timer = setInterval(draw, 1000 / fps)
  const stream = canvas.captureStream(fps)
  return { stream, track: stream.getVideoTracks()[0], stop: () => { clearInterval(timer); stream.getTracks().forEach(t => t.stop()) } }
}

type MstpCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

// Continuous frame counter on a track (reads via MediaStreamTrackProcessor).
export function frameCounter(track: MediaStreamTrack) {
  const Proc = (self as unknown as { MediaStreamTrackProcessor: MstpCtor }).MediaStreamTrackProcessor
  const reader = new Proc({ track }).readable.getReader()
  const state = { frames: 0, done: false, lastAt: 0 }
  ;(async () => {
    for (;;) {
      const { value, done } = await reader.read().catch(() => ({ value: undefined, done: true }))
      if (done || !value) { state.done = true; break }
      value.close()
      state.frames++
      state.lastAt = performance.now()
    }
  })()
  return {
    get frames() { return state.frames },
    get done() { return state.done },
    // Frames that arrive in the next `ms`.
    async during(ms: number): Promise<number> { const a = state.frames; await sleep(ms); return state.frames - a },
    // Longest gap between frames over the next `ms` (ms).
    async maxGap(ms: number): Promise<number> {
      const end = performance.now() + ms
      let last = performance.now(), gap = 0, seen = state.frames
      while (performance.now() < end) {
        await sleep(10)
        if (state.frames !== seen) { seen = state.frames; last = performance.now() }
        gap = Math.max(gap, performance.now() - last)
      }
      return gap
    },
    stop: () => { void reader.cancel().catch(() => {}) },
  }
}

let realFetch: typeof fetch | null = null
// Serve the fixture for every model_*.bin; optional extra routes.
export async function stubWeights(extra: Record<string, () => Response> = {}): Promise<void> {
  realFetch ??= globalThis.fetch.bind(globalThis)
  const bin = await (await realFetch(binUrl)).arrayBuffer()
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    for (const [pattern, respond] of Object.entries(extra)) if (url.includes(pattern)) return respond()
    if (/model_[a-z]+\.f16\.bin$/.test(url)) return new Response(null, { status: 404 })
    if (/model_[a-z]+\.bin$/.test(url)) return new Response(bin.slice(0))
    return realFetch!(input, init)
  }) as typeof fetch
}

export function restoreFetch(): void {
  if (realFetch) globalThis.fetch = realFetch
  resetTuning()
}

export function errorLog() {
  const errors: PipelineError[] = []
  return {
    errors,
    onError: (e: PipelineError) => { errors.push(e) },
    types: () => errors.map(e => e.type),
    async waitFor(type: string, ms = 10_000): Promise<PipelineError> {
      const end = performance.now() + ms
      for (;;) {
        const e = errors.find(x => x.type === type)
        if (e) return e
        if (performance.now() > end) throw new Error(`timed out waiting for onError '${type}'; got [${errors.map(x => x.type).join(', ')}]`)
        await sleep(25)
      }
    },
  }
}

export async function waitUntil(cond: () => boolean, ms = 10_000, what = 'condition'): Promise<void> {
  const end = performance.now() + ms
  while (!cond()) {
    if (performance.now() > end) throw new Error(`timed out waiting for ${what}`)
    await sleep(25)
  }
}
