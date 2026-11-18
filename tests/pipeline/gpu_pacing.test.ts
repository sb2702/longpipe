import { describe, it, expect, afterEach, beforeEach } from 'vitest'
import { Pipeline } from '~/pipeline/index'
import { TUNING } from '~/pipeline/tuning'
import { stubWeights, restoreFetch, SMALL, sleep } from '../helpers/pipeline'

// GPU pacing on the outputs without consumer backpressure (bitmap-shuttle,
// transfer-capture): with a GPU slower than the camera, the output must stay
// close to real time (stale input frames drop) instead of falling further
// behind. A heavy output size on headless Chromium's software GPU stands in for
// a slow device. (Measured 2026-11: headless Chromium/SwiftShader and Firefox/
// llvmpipe don't queue much even without pacing, as their software GL blocks on
// its own; this guards that pacing keeps latency bounded and doesn't cost
// throughput. Run in Firefox with a vitest config whose instance is firefox +
// 'webgl.force-enabled'.)
//
// Each camera frame is one flat colour that encodes the time it was drawn
// (flat stays flat through blur and compositing): R = 16 ms steps, G = 4 s steps.
// Output frames are decoded back and their age measured.

const STEP = 16
function colourFor(t: number): [number, number, number] {
  const q = Math.floor(t / STEP)
  return [q % 256, Math.floor(q / 256) % 256, 128]
}
function timeFor(r: number, g: number, now: number): number {
  // Nearest candidate to `now` (absorbs a carry between R and G).
  const base = (g * 256 + r) * STEP
  const period = 256 * 256 * STEP
  let best = base
  for (let k = -2; k <= 2; k++) {
    const c = base + Math.round((now - base) / period + k) * period
    if (Math.abs(now - c) < Math.abs(now - best)) best = c
  }
  return best
}

function timedCamera(w: number, h: number, fps = 30) {
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')!
  const draw = () => {
    const [r, g, b] = colourFor(performance.now())
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(0, 0, w, h)
  }
  draw()
  const timer = setInterval(draw, 1000 / fps)
  const stream = canvas.captureStream(fps)
  return { stream, stop: () => { clearInterval(timer); stream.getTracks().forEach(t => t.stop()) } }
}

// Ages (ms) of output frames over `ms`, read through a <video> (rVFC), which
// works on every engine.
async function outputAges(track: MediaStreamTrack, ms: number): Promise<number[]> {
  const video = document.createElement('video')
  video.muted = true
  video.srcObject = new MediaStream([track])
  await video.play()
  const c = document.createElement('canvas')
  c.width = c.height = 8
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  const ages: number[] = []
  await new Promise<void>((resolve) => {
    const end = performance.now() + ms
    const onFrame = () => {
      const now = performance.now()
      if (now > end) { resolve(); return }
      ctx.drawImage(video, video.videoWidth / 2 - 4, video.videoHeight / 2 - 4, 8, 8, 0, 0, 8, 8)
      const [r, g] = ctx.getImageData(4, 4, 1, 1).data
      ages.push(now - timeFor(r, g, now))
      video.requestVideoFrameCallback(onFrame)
    }
    video.requestVideoFrameCallback(onFrame)
  })
  video.pause()
  video.srcObject = null
  return ages
}

const median = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] ?? NaN

let cleanup: (() => void)[] = []
beforeEach(async () => { await stubWeights() })
afterEach(() => { for (const f of cleanup.reverse()) { try { f() } catch {} } cleanup = []; restoreFetch() })

const CASES = [
  { output: 'bitmap-shuttle',   backend: 'webgl'  },
  { output: 'bitmap-shuttle',   backend: 'webgpu' },
  { output: 'transfer-capture', backend: 'webgl'  },
] as const

describe.each(CASES)('GPU pacing ($output, $backend)', ({ output, backend }) => {
  const firefox = navigator.userAgent.includes('Firefox')
  it.skipIf(firefox && output === 'transfer-capture')('latency stays bounded when the GPU is slower than the camera', async () => {
    TUNING.forceTopology = { input: 'rvfc-postmessage', output }
    TUNING.forceBackend  = backend
    // A cheap camera, a heavy output: the GPU (not the camera) is the bottleneck.
    const W = 2560
    const cam = timedCamera(640, 360)
    const p = new Pipeline(cam.stream, { preset: SMALL, background: 'blur', outputResolution: { w: W, h: Math.round(W * 9 / 16) } })
    cleanup.push(() => p.destroy(), cam.stop)
    await p.ready
    await sleep(1000)
    const early = await outputAges(p.stream.getVideoTracks()[0], 3000)
    await sleep(4000)
    const late = await outputAges(p.stream.getVideoTracks()[0], 3000)
    const stats = await p.getStats()
    console.log(`[gpu pacing] ${output}/${backend}: fps ${stats.fps}, input ${stats.inputFps}, ` +
      `age median ${median(early).toFixed(0)} ms (s1-4) → ${median(late).toFixed(0)} ms (s8-11), ` +
      `max ${Math.max(...late).toFixed(0)} ms, frames ${early.length}/${late.length}`)
    expect(late.length).toBeGreaterThan(5)
    // Pacing must not starve the output (e.g. a fence that never reads as signalled).
    expect(stats.fps).toBeGreaterThan(stats.inputFps * 0.5)
    // Bounded: doesn't grow over time, and stays well under a second.
    expect(median(late)).toBeLessThan(Math.max(500, median(early) * 1.5 + 100))
  }, 60_000)
})
