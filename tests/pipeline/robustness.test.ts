import { describe, it, expect, afterEach, beforeEach } from 'vitest'
import { Pipeline } from '~/pipeline/index'
import { TUNING } from '~/pipeline/tuning'
import { LongpipeError, prefixMessage } from '~/pipeline/errors'
import { PRESETS, resolveNamedPreset, type ModelName } from '~/pipeline/presets'
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

  // The device self-test's sequence: a GPU loss, then a crash of the running
  // worker. The crash says nothing about WebGPU, so it must not count.
  it('a crash of a running worker does not count: GPU loss + crash stays on WebGPU', async () => {
    TUNING.watchdogMs = 1500
    const { pipeline, log, recovered } = start()
    await pipeline.ready
    await pipeline._debug('lose-backend').catch(() => {})
    await waitUntil(() => recovered.length === 1, 10_000, 'recovery after GPU loss')
    expect(pipeline.backend).toBe('webgpu')
    await pipeline._debug('terminate')
    await log.waitFor('worker-crash', 15_000)
    await waitUntil(() => recovered.length === 2, 10_000, 'recovery after crash')
    expect(recovered[1]).toMatchObject({ reason: 'worker-crash', backend: 'webgpu' })
    expect(pipeline.backend).toBe('webgpu')
  })

  // A worker dying during start-up (Firefox's pre-init WebGPU crash) does count.
  it('a crash during start-up counts: start-up crash + GPU loss moves to WebGL2', async () => {
    TUNING.initTimeoutMs = 1500
    const { pipeline, log, recovered } = start()
    await pipeline._debug('terminate')   // before init answers
    await log.waitFor('worker-crash', 10_000)
    await pipeline.ready
    expect(pipeline.backend).toBe('webgpu')
    await pipeline._debug('lose-backend').catch(() => {})
    await waitUntil(() => recovered.length >= 1 && pipeline.backend === 'webgl2', 15_000, 'recovery on webgl2')
    expect(pipeline.backend).toBe('webgl2')
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
    // Detection takes watchdogMs + up to one 1 s watchdog tick, plus any time
    // the watchdog is paused by a heavy command still in flight after ready
    // (busyCall): under a loaded suite that can exceed 6 s. A generous bound
    // costs nothing when it passes.
    const e = await log.waitFor('worker-crash', 15_000)
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

describe('tier names as presets', () => {
  it('resolve to the shipped preset for that tier', () => {
    for (const p of PRESETS) expect(resolveNamedPreset(p.model)).toBe(p)
    expect(resolveNamedPreset('fast')).toBe(PRESETS[0])
    expect(resolveNamedPreset('nope' as never)).toBeNull()
    expect(resolveNamedPreset('toString' as never)).toBeNull()
  })

  it("preset: 'small' at construction and setPreset('small') at runtime; ready.preset round-trips", async () => {
    const { pipeline, log } = start({ preset: 'small' })
    const info = await pipeline.ready
    expect(info.preset).toBe('small')
    await pipeline.setPreset(info.preset as ModelName)
    expect((await pipeline.getStats()).preset).toBe('small')
    expect(log.errors).toEqual([])
  })
})

describe('enabled: false', () => {
  function countModelFetches() {
    const inner = globalThis.fetch
    const urls: string[] = []
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (/model_[a-z]+(\.f16)?\.bin$/.test(url)) urls.push(url)
      return inner(input, init)
    }) as typeof fetch
    return urls
  }

  it('fetches no weights until setEnabled(true), then loads and runs the effect', async () => {
    const fetched = countModelFetches()
    const { pipeline, out, log } = start({ enabled: false })
    const info = await pipeline.ready
    expect(info).toMatchObject({ backend: 'webgpu', preset: null })
    expect(fetched).toEqual([])
    expect(await out.during(800)).toBeGreaterThan(10)
    expect((await pipeline.getStats()).enabled).toBe(false)

    await pipeline.setEnabled(true)
    expect(fetched.length).toBeGreaterThan(0)
    const stats = await pipeline.getStats()
    expect(stats).toMatchObject({ enabled: true, preset: 'small', status: 'running' })
    expect(await out.during(800)).toBeGreaterThan(10)
    expect(log.errors).toEqual([])
  })

  it('setPreset while disabled records the tier without fetching; setEnabled(true) loads it', async () => {
    const fetched = countModelFetches()
    const { pipeline } = start({ enabled: false, preset: 'auto', adaptive: false, backend: 'webgpu' })
    // 'auto' would pick some tier; pin 'small' (the only fixture) before enabling.
    await pipeline.ready.catch(() => {})
    await pipeline.setPreset('small')
    expect(fetched).toEqual([])
    await pipeline.setEnabled(true)
    expect((await pipeline.getStats()).preset).toBe('small')
  })
})

describe('typed control-call errors', () => {
  it('Longpipe-only calls under a MediaPipe model reject with a LongpipeError and fire onError', async () => {
    TUNING.forceBackend = 'none'
    const { pipeline, log } = start({ model: 'mediapipe-selfie', preset: undefined })
    for (const call of [
      () => pipeline.setTouchup({}),
      () => pipeline.setReframe(true),
      () => pipeline.reframe(),
      () => pipeline.setPreset('fast'),
    ]) {
      const err = await call().then(() => null, (e: unknown) => e)
      expect(err).toBeInstanceOf(LongpipeError)
      expect(err).toMatchObject({ type: 'control-failed', fatal: false })
    }
    expect(log.types().filter(t => t === 'control-failed')).toHaveLength(4)
  })

  it('constructor option checks throw a LongpipeError', () => {
    const cam = syntheticCamera(30)
    cleanup.push(() => cam.stop())
    expect(() => new Pipeline(cam.stream, { model: 'mediapipe-selfie', touchup: {} }))
      .toThrow(expect.objectContaining({ name: 'LongpipeError', type: 'init-failed' }))
  })

  it('attachPreview on an already-transferred canvas and setInputStream with no video reject typed', async () => {
    const { pipeline } = start()
    await pipeline.ready
    const canvas = document.createElement('canvas')
    await pipeline.attachPreview(canvas)
    await expect(pipeline.attachPreview(canvas)).rejects.toBeInstanceOf(LongpipeError)
    await expect(pipeline.setInputStream(new MediaStream())).rejects.toMatchObject({ type: 'control-failed' })
  })
})

describe('error messages', () => {
  it('prefixes once', async () => {
    expect(prefixMessage('init failed', 'init failed: x')).toBe('init failed: x')
    expect(prefixMessage('init failed', 'x')).toBe('init failed: x')
    const { pipeline } = start({ preset: 'nope' })
    const err = await pipeline.ready.then(() => null, (e: unknown) => e as LongpipeError)
    expect(err).toMatchObject({ type: 'init-failed' })
    expect(err!.message).toMatch(/^init failed: resolve_preset: unknown preset name 'nope'/)
    expect(err!.message).not.toMatch(/init failed: init failed/)
  })
})
