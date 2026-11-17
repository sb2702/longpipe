import { describe, it, expect, afterEach, beforeEach } from 'vitest'
import { Pipeline } from '~/pipeline/index'
import { TUNING } from '~/pipeline/tuning'
import { syntheticCamera, frameCounter, stubWeights, restoreFetch, errorLog, SMALL, sleep, waitUntil } from '../helpers/pipeline'

let cleanup: (() => void)[] = []
beforeEach(async () => { await stubWeights() })
afterEach(() => { for (const f of cleanup.reverse()) { try { f() } catch {} } cleanup = []; restoreFetch() })

function start(opts: Record<string, unknown> = {}) {
  const cam = syntheticCamera(30)
  const log = errorLog()
  const recovered: unknown[] = []
  const pipeline = new Pipeline(cam.stream, {
    preset: SMALL, background: 'blur', onError: log.onError,
    onRecovered: (i) => recovered.push(i), ...opts,
  })
  const out = frameCounter(pipeline.stream.getVideoTracks()[0])
  cleanup.push(() => out.stop(), () => pipeline.destroy(), () => cam.stop())
  return { cam, log, pipeline, out, recovered }
}

describe('EffectsPipeline happy path', () => {
  it('reaches ready on WebGPU and reports the backend', async () => {
    const { pipeline, out, log } = start()
    const info = await pipeline.ready
    expect(info.backend).toBe('webgpu')
    expect(pipeline.backend).toBe('webgpu')
    const stats = await pipeline.getStats()
    expect(stats.backend).toBe('webgpu')
    expect(stats.status).toBe('running')
    expect(await out.during(1000)).toBeGreaterThan(15)
    expect(log.errors).toEqual([])
  })
})

const TOPOLOGIES = [
  { name: 'mstg',             topology: { input: 'mstp', output: 'mstg' } },
  { name: 'bitmap-shuttle',   topology: { input: 'rvfc-postmessage', output: 'bitmap-shuttle' } },
  { name: 'transfer-capture', topology: { input: 'rvfc-postmessage', output: 'transfer-capture' } },
] as const

describe.each(TOPOLOGIES)('no GPU at startup ($name)', ({ topology }) => {
  it('outputs raw camera frames, fires a non-fatal no-backend error, reports backend none', async () => {
    TUNING.forceTopology = { ...topology }
    TUNING.forceBackend  = 'none'
    const { pipeline, out, log } = start()
    const info = await pipeline.ready
    expect(info.backend).toBe('none')
    expect(pipeline.backend).toBe('none')
    const e = await log.waitFor('no-backend')
    expect(e.fatal).toBe(false)
    expect(e.recoverable).toBe(true)
    expect(await out.during(1000)).toBeGreaterThan(10)
    const stats = await pipeline.getStats()
    expect(stats.backend).toBe('none')
    expect(stats.status).toBe('passthrough')
    // Control calls reject clearly (typed) instead of hanging.
    await expect(pipeline.setBackground('none')).rejects.toMatchObject({ type: 'no-backend' })
  })
})

describe('bad background', () => {
  it('at construction: starts with none, non-fatal background-load, frames flow', async () => {
    const { pipeline, out, log } = start({ background: '/does-not-exist.png' })
    const info = await pipeline.ready
    expect(info.backend).toBe('webgpu')
    const e = await log.waitFor('background-load')
    expect(e.fatal).toBe(false)
    expect(await out.during(800)).toBeGreaterThan(10)
  })

  it('at runtime: the call rejects typed, previous effect kept, ready unaffected', async () => {
    const { pipeline, out, log } = start()
    await pipeline.ready
    await expect(pipeline.setBackground('/nope.jpg')).rejects.toMatchObject({ type: 'background-load', fatal: false })
    expect(log.types()).toContain('background-load')
    // A blob that isn't an image (decode failure) is handled the same way.
    await expect(pipeline.setBackground({ image: { data: new Uint8Array([1, 2, 3]), type: 'image/png' } }))
      .rejects.toMatchObject({ type: 'background-load' })
    expect(await out.during(800)).toBeGreaterThan(10)
    expect((await pipeline.getStats()).status).toBe('running')
  })
})

describe.each(['webgpu', 'webgl'] as const)('GPU backend lost (%s)', (backend) => {
  it('keeps frames flowing on the same track, recovers, fires onRecovered', async () => {
    TUNING.forceBackend = backend
    const { pipeline, out, log, recovered } = start()
    await pipeline.ready
    const track = pipeline.stream.getVideoTracks()[0]
    const gapP = out.maxGap(4000)
    await pipeline._debug('lose-backend').catch(() => {})
    const e = await log.waitFor('backend-lost')
    expect(e.fatal).toBe(false)
    await waitUntil(() => recovered.length > 0, 10_000, 'onRecovered')
    expect(recovered[0]).toMatchObject({ reason: 'backend-lost', backend: backend === 'webgl' ? 'webgl2' : 'webgpu' })
    // Raw passthrough bridged the gap: no long freeze on the output track.
    expect(await gapP).toBeLessThan(700)
    expect(pipeline.stream.getVideoTracks()[0]).toBe(track)
    expect((await pipeline.getStats()).status).toBe('running')
    expect(await out.during(800)).toBeGreaterThan(10)
  })
})

describe('second WebGPU loss falls back to WebGL2', () => {
  it('recovers on webgl2 after losing WebGPU twice', async () => {
    const { pipeline, log, recovered } = start()
    await pipeline.ready
    await pipeline._debug('lose-backend').catch(() => {})
    await waitUntil(() => recovered.length === 1, 10_000, 'first recovery')
    await pipeline._debug('lose-backend').catch(() => {})
    await waitUntil(() => recovered.length === 2, 10_000, 'second recovery')
    expect(pipeline.backend).toBe('webgl2')
    expect(log.types().filter(t => t === 'backend-lost').length).toBe(2)
  })
})

describe.each(TOPOLOGIES)('worker crash ($name)', ({ name, topology }) => {
  it('watchdog detects a killed worker, raw frames bridge, worker restarts', async () => {
    TUNING.forceTopology = { ...topology }
    TUNING.watchdogMs = 1500
    const changes: { videoTrack: MediaStreamTrack }[] = []
    const { pipeline, out, log, recovered } = start({ onTrackChanged: (c: { videoTrack: MediaStreamTrack }) => changes.push(c) })
    await pipeline.ready
    const track = pipeline.stream.getVideoTracks()[0]
    await pipeline._debug('terminate')
    const e = await log.waitFor('worker-crash', 6000)
    expect(e.fatal).toBe(false)
    await waitUntil(() => recovered.length > 0, 10_000, 'onRecovered')
    expect(recovered[0]).toMatchObject({ reason: 'worker-crash' })
    if (name === 'transfer-capture') {
      // The transferred canvas died with the worker: documented track change.
      expect(changes.length).toBe(1)
      expect(pipeline.stream.getVideoTracks()[0]).toBe(changes[0].videoTrack)
      const out2 = frameCounter(changes[0].videoTrack)
      cleanup.push(() => out2.stop())
      expect(await out2.during(1000)).toBeGreaterThan(10)
    } else {
      expect(changes.length).toBe(0)
      expect(pipeline.stream.getVideoTracks()[0]).toBe(track)
      expect(await out.during(1000)).toBeGreaterThan(10)
    }
    expect((await pipeline.getStats()).restarts).toBe(1)
  })
})

describe('worker crash during recovery window', () => {
  it('raw frames keep flowing on mstg while the worker is down', async () => {
    TUNING.watchdogMs = 1000
    TUNING.restartBackoffMs = [2000]
    const { pipeline, out, log } = start()
    await pipeline.ready
    await pipeline._debug('terminate')
    await log.waitFor('worker-crash', 5000)
    // Between detection and restart (2 s backoff) the relay serves raw frames.
    expect(await out.during(1000)).toBeGreaterThan(15)
    expect((await pipeline.getStats()).status).toBe('recovering')
  })

  it('gives up after maxRestarts with a fatal recovery-failed, output stays raw', async () => {
    TUNING.watchdogMs = 800
    TUNING.maxRestarts = 0
    const { pipeline, out, log } = start()
    await pipeline.ready
    await pipeline._debug('terminate')
    const e = await log.waitFor('recovery-failed', 5000)
    expect(e.fatal).toBe(true)
    expect(await out.during(800)).toBeGreaterThan(10)
    await expect(pipeline.setEnabled(false)).rejects.toMatchObject({ type: 'recovery-failed' })
  })
})

describe('camera track ended', () => {
  it('fires camera-ended (fatal) and ends the output track', async () => {
    const { pipeline, cam, log } = start()
    await pipeline.ready
    const outTrack = pipeline.stream.getVideoTracks()[0]
    cam.track.stop()   // app stop(): no 'ended' event; detected by poll / worker
    const e = await log.waitFor('camera-ended', 4000)
    expect(e.fatal).toBe(true)
    await waitUntil(() => outTrack.readyState === 'ended', 2000, 'output track ended')
    await expect(pipeline.setEnabled(true)).rejects.toMatchObject({ type: 'camera-ended' })
  })

  it('setInputStream switches cameras on the same output track', async () => {
    const { pipeline, out, cam, log } = start()
    await pipeline.ready
    const track = pipeline.stream.getVideoTracks()[0]
    const cam2 = syntheticCamera(30)
    cleanup.push(() => cam2.stop())
    await pipeline.setInputStream(cam2.stream)
    cam.track.stop()
    await sleep(1500)
    expect(log.types()).not.toContain('camera-ended')
    expect(pipeline.stream.getVideoTracks()[0]).toBe(track)
    expect(await out.during(1000)).toBeGreaterThan(10)
  })
})

describe('control calls', () => {
  it('a failing call rejects typed, fires non-fatal onError, pipeline keeps running', async () => {
    const { pipeline, out, log } = start()
    await pipeline.ready
    await expect(pipeline.setPreset({ ...SMALL, model: 'nope' as never })).rejects.toMatchObject({ type: 'control-failed', fatal: false })
    await expect(pipeline._debug('throw')).rejects.toMatchObject({ type: 'control-failed' })
    expect(log.types()).toContain('control-failed')
    expect((await pipeline.getStats()).status).toBe('running')
    expect(await out.during(800)).toBeGreaterThan(10)
  })

  it('a call that never answers times out instead of hanging', async () => {
    TUNING.controlTimeoutMs = 300
    const { pipeline } = start()
    await pipeline.ready
    await expect(pipeline._debug('hang')).rejects.toMatchObject({ type: 'control-timeout' })
  })

  it('calls made before ready queue and apply; a later failure never rejects ready', async () => {
    const { pipeline, log } = start()
    const early = pipeline.setEnabled(false)
    const earlyBad = pipeline.setBackground('/nope.png')
    await expect(early).resolves.toBeUndefined()
    await expect(earlyBad).rejects.toMatchObject({ type: 'background-load' })
    await expect(pipeline.ready).resolves.toMatchObject({ backend: 'webgpu' })
    expect((await pipeline.getStats()).enabled).toBe(false)
    expect(log.errors.every(e => !e.fatal)).toBe(true)
  })

  it('setPreset fetches its own weights and stops adaptive', async () => {
    const { pipeline } = start()
    await pipeline.ready
    // (The fixture only has 'small' weights, so autotune can't run here; plant
    // a stand-in controller to check setPreset stops it.)
    let stopped = false
    ;(pipeline as unknown as { adaptive: unknown }).adaptive = { stop: () => { stopped = true } }
    await pipeline.setPreset(SMALL)   // no weights argument
    expect(stopped).toBe(true)
    expect((pipeline as unknown as { adaptive: unknown }).adaptive).toBeNull()
    expect((await pipeline.getStats()).preset).toBe('small')
  })
})
