import { describe, it, expect, afterEach } from 'vitest'
import { AdaptiveController } from '~/pipeline/adaptive'
import { InputRateMeter } from '~/pipeline/input_rate'
import { setupPostMessageInput } from '~/pipeline/transports/input_postmessage'
import type { RendererStats } from '~/pipeline/messages'
import { syntheticCamera, sleep } from '../helpers/pipeline'

const stats = (fps: number, modelMs = 20): RendererStats => ({
  fps, modelFps: fps / 2, modelMs, displayMs: 1, skipped: 0, preset: 'medium', skipFrames: 1,
  enabled: true, inputPath: 'mstp', outputPath: 'mstg', backend: 'webgpu',
})

// Drive the controller with a fake clock: one tick every 2 s for `seconds`.
async function run(inputFps: number | null, renderedFps: number, seconds = 30) {
  const swaps: string[] = []
  const t0 = 1_000_000
  const c = new AdaptiveController({
    backendKind:  'webgpu',
    initialModel: 'medium',
    getInputFps:  () => inputFps,
    fetchWeights: async () => new ArrayBuffer(8),
    getStats:     async () => stats(renderedFps),
    swapPreset:   async (p) => { swaps.push(typeof p === 'string' ? p : p.model) },
  }, t0)
  for (let t = 0; t <= seconds * 1000; t += 2000) await c.tick(t0 + t)
  return swaps
}

describe('AdaptiveController target = min(measured input fps, 30)', () => {
  it('a 15 fps camera rendering 15 fps is on target: no downgrade', async () => {
    expect(await run(15, 15)).toEqual([])
  })

  it('a 30 fps camera rendering 15 fps is overloaded: downgrades', async () => {
    expect(await run(30, 15)).toEqual(['small'])
  })

  it('unknown input rate falls back to the 30 fps assumption (0.0.18 behaviour)', async () => {
    expect(await run(null, 15)).toEqual(['small'])
  })

  it('a 15 fps camera rendering 8 fps (below 2/3 of target) still downgrades', async () => {
    expect(await run(15, 8)).toEqual(['small'])
  })

  it('targetFps is capped at 30 and clamped for near-zero rates', () => {
    const mk = (fps: number) => new AdaptiveController({
      backendKind: 'webgl', initialModel: 'small', getInputFps: () => fps,
      fetchWeights: async () => new ArrayBuffer(0), getStats: async () => stats(30), swapPreset: async () => {},
    })
    expect(mk(60).targetFps()).toBe(30)
    expect(mk(15).targetFps()).toBe(15)
    expect(mk(0.5).targetFps()).toBe(5)
  })
})

describe('InputRateMeter on a 15 fps synthetic source', () => {
  let stop: (() => void)[] = []
  afterEach(() => { stop.forEach(f => f()); stop = [] })

  it('measures ~15 fps from rVFC presented frames', async () => {
    const cam = syntheticCamera(15)
    const input = setupPostMessageInput(cam.stream)
    stop.push(cam.stop, input.cleanup)
    const meter = new InputRateMeter(() => cam.track, () => input.framesIn)
    for (let i = 0; i < 6; i++) { meter.sample(); await sleep(400) }
    meter.sample()
    const fps = meter.fps()!
    expect(fps).toBeGreaterThan(11)
    expect(fps).toBeLessThan(19)
  })

  it('falls back to the track settings frame rate when it cannot count', () => {
    const cam = syntheticCamera(15)
    stop.push(cam.stop)
    const meter = new InputRateMeter(() => cam.track, () => undefined)
    meter.sample()
    expect(meter.fps()).toBe(15)
  })
})
