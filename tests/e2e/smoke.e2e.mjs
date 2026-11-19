// Cross-browser smoke tests for EffectsPipeline inside a real WebRTC call.
//
// Each test opens the vanilla-WebRTC example with a deterministic input (the
// pinned clip of a person drawn into a canvas + a WebAudio tone and noise),
// runs pipeline.stream through a pc1 → pc2 loopback, and checks what pc2
// RECEIVED: decoded frames and pixels of the remote <video>, received audio.
//
// Every scenario runs with backend 'auto', 'webgpu' and 'webgl2' (the SDK's
// `backend` option). A forced backend the browser can't provide is SKIPPED
// with the reason; the backend each run actually used is recorded and lands in
// results/summary.md.
import { test as base, expect } from '@playwright/test'

const GREEN = [0, 255, 0]
const CORNERS = { tl: [0.04, 0.06], tr: [0.96, 0.06], bl: [0.04, 0.94], br: [0.96, 0.94] }
const SUBJECT = { face: [0.48, 0.35], chest: [0.5, 0.82] }
const BACKENDS = ['auto', 'webgpu', 'webgl2']
const UNREACHABLE = 'http://127.0.0.1:9/'   // connection refused: weights + denoise wasm fail
const RAW_TONE_RMS = 0.05                     // the clip input's tone + noise is ~0.15 RMS

const dist = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))

// ── Fixture: one example page per test, torn down with a console-error check ──

// Browsers whose CI runner can't connect the in-page WebRTC call (E2E_NO_RTC=firefox):
// check MediaStream in → MediaStream out on the local pipeline.stream instead.
let noRtc = false

const test = base.extend({
  call: async ({ page }, use, testInfo) => {
    noRtc = (process.env.E2E_NO_RTC ?? '').split(',').includes(testInfo.project.name)
    if (noRtc) testInfo.annotations.push({ type: 'scope', description: 'no WebRTC: checked pipeline.stream locally' })
    const consoleErrors = []
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()) })
    page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))

    const call = {
      consoleErrors,
      async start(params) {
        const q = new URLSearchParams({ input: 'clip', preset: 'fast', weights: 'local', denoiseModel: 'rnnoise', ...params })
        testInfo.annotations.push({ type: 'requested', description: params.backend ?? 'auto' })
        await page.goto(`/?${q}`)
        await page.click('#start')   // a real gesture: AudioContexts may start
        await expect.poll(() => page.evaluate(() => window.example.state.running), { timeout: 30_000 }).toBe(true)
      },
      // Wait until `ready` settled (resolved or rejected). Returns the state.
      async settled(timeout = 60_000) {
        try {
          await expect.poll(() => page.evaluate(() => {
            const s = window.example.state
            return !!(s.ready || s.readyError)
          }), { timeout, intervals: [250] }).toBe(true)
        } catch (e) {
          // Say where it got stuck: the example's state, the SDK's stats and the audio context.
          const dump = await page.evaluate(async () => {
            const s = window.example.state
            const stats = await window.example.pipeline?.getStats?.().catch((err) => String(err))
            return JSON.stringify({ running: s.running, config: s.config, errors: s.errors, stats, audio: window.example.input?.audioContext?.state })
          }).catch((err) => `state unavailable: ${err.message}`)
          throw new Error(`ready never settled after ${timeout} ms. State: ${dump}`)
        }
        return this.state()
      },
      state: () => page.evaluate(async () => {
        await window.example.refreshStats()
        return JSON.parse(JSON.stringify(window.example.state))
      }),
      sample: (points) => page.evaluate(([p, id]) => window.example.sampleRemote(p, id), [points, noRtc ? 'local' : 'remote']),
      levels: () => page.evaluate(() => window.example.audioLevels()),
      eval: (fn, arg) => page.evaluate(fn, arg),
      record(actual) { testInfo.annotations.push({ type: 'backend', description: actual }) },
    }
    await use(call)

    // Teardown: destroy() (via the example's stop) must be clean.
    const before = consoleErrors.length
    const ended = await page.evaluate(async () => {
      if (!window.example) return []   // test skipped before the example loaded
      const tracks = window.example.pipeline?.stream.getTracks() ?? []
      await window.example.stop()
      await new Promise((r) => setTimeout(r, 1500))
      return tracks.map((t) => `${t.kind}:${t.readyState}`)
    }).catch((e) => [`stop failed: ${e.message}`])
    const after = consoleErrors.slice(before)
    testInfo.annotations.push({ type: 'teardown', description: `tracks after destroy: ${ended.join(', ') || 'none'}` })
    expect(after, 'console errors after destroy()').toEqual([])
    // The example always builds the denoiser, so both output tracks are the
    // SDK's own and must end. (With audio: 'passthrough' the audio track is the
    // app's input track, which destroy() rightly leaves alone.)
    if (!ended[0]?.startsWith('stop failed')) expect(ended, 'output tracks after destroy()').toEqual(ended.map((t) => t.replace(/:.*/, ':ended')))
  },
})

test.describe.configure({ mode: 'parallel' })
test.beforeEach(({}, testInfo) => {
  testInfo.annotations.push({ type: 'scenario', description: testInfo.title.split(' [')[0] })
})

// Record the backend the pipeline actually got, and skip when a FORCED one
// wasn't available (the SDK fell back, as it should; there's nothing to test).
const NO_GPU_REASON = 'no GPU backend reachable from the SDK worker here (Playwright\'s WPE WebKit has no WebGL2/WebGPU in workers); ' +
  'the SDK failed open to raw video, which the no-GPU test covers'

function checkBackend(call, s, requested, { allowNone = false } = {}) {
  const actual = s.ready?.backend ?? s.stats?.backend ?? 'none'
  call.record(actual)
  const lost = s.errors.filter((e) => e.type === 'backend-lost').length
  if (requested === 'webgpu' && actual !== 'webgpu') {
    test.skip(true, lost
      ? `WebGPU device lost at startup ${lost}x (headless Chromium destroys it on the first canvas present); SDK fell back to ${actual}`
      : `no WebGPU adapter in this browser/headless setup; SDK fell back to ${actual}`)
  }
  if (requested === 'webgl2' && actual !== 'webgl2') test.skip(true, `no WebGL2 (EXT_color_buffer_float) here; got ${actual}`)
  if (requested === 'auto' && actual === 'none' && !allowNone) {
    expect(s.errors.find((e) => e.type === 'no-backend')?.fatal, 'no GPU must be a non-fatal no-backend error').toBe(false)
    test.skip(true, NO_GPU_REASON)
  }
  if (requested === 'auto' && !allowNone) expect(['webgpu', 'webgl2'], `effect backend (errors: ${JSON.stringify(s.errors)})`).toContain(actual)
  return actual
}

// "Flowing", not "fast": WebKit's software path here delivers ~3-5 fps.
async function expectFramesFlow(call, ms = 3000) {
  if (noRtc) {
    const n = await call.eval((t) => window.example.countLocalFrames(t), ms)
    expect(n, `pipeline.stream frames in ${ms} ms`).toBeGreaterThanOrEqual(3)
    const s = await call.state()
    s.measuredRemoteFps = n / (ms / 1000)
    return s
  }
  const a = await call.state()
  await new Promise((r) => setTimeout(r, ms))
  const b = await call.state()
  const decoded = (b.rtp.video.framesDecoded ?? 0) - (a.rtp.video.framesDecoded ?? 0)
  let why = ''
  if (decoded < 3) {
    // Say why: connection state, ICE candidates and the local video track.
    why = await call.eval(async () => {
      const c = window.example.call
      if (!c) return 'no call'
      const cands = []
      for (const [name, pc] of [['pc1', c.pc1], ['pc2', c.pc2]]) {
        const st = await pc.getStats()
        st.forEach((r) => { if (r.type === 'local-candidate') cands.push(`${name}:${r.candidateType}/${r.protocol}/${r.address ?? r.ip}`) })
      }
      const t = c.senders?.video?.track
      return JSON.stringify({
        pc1: [c.pc1.connectionState, c.pc1.iceConnectionState, c.pc1.iceGatheringState],
        pc2: [c.pc2.connectionState, c.pc2.iceConnectionState, c.pc2.iceGatheringState],
        cands, videoTrack: t ? [t.readyState, t.muted, t.label] : null,
      })
    }).catch((e) => `diag failed: ${e.message}`)
  }
  expect(decoded, `remote frames decoded in ${ms} ms ${why}`).toBeGreaterThanOrEqual(3)
  b.measuredRemoteFps = decoded / (ms / 1000)
  return b
}

async function expectBackground(call, color = GREEN) {
  await expect.poll(async () => {
    const s = await call.sample({ ...CORNERS, ...SUBJECT })
    if (!s) return 'no remote frame yet'
    const bad = Object.keys(CORNERS).filter((k) => dist(s.patches[k], color) > 60)
    const subj = Object.keys(SUBJECT).filter((k) => dist(s.patches[k], color) < 100)
    return bad.length || subj.length ? `corners off-colour: ${bad} / subject replaced: ${subj} ${JSON.stringify(s.patches)}` : 'ok'
  }, { timeout: 30_000, intervals: [500] }).toBe('ok')
}

async function expectRawVideo(call, color = GREEN) {
  const s = await call.sample(CORNERS)
  expect(s, 'remote frame').not.toBeNull()
  const coloured = Object.keys(CORNERS).filter((k) => dist(s.patches[k], color) < 60)
  expect(coloured, `corners must be the raw clip, not the background colour: ${JSON.stringify(s.patches)}`).toEqual([])
}

async function expectAudioReceived(call) {
  if (noRtc) {
    const lv = await call.levels()
    expect(lv.local, 'pipeline.stream audio RMS (not silence)').toBeGreaterThan(1e-5)
    lv.remote = lv.local   // later checks compare "what would be sent"
    return lv
  }
  const a = await call.state()
  await new Promise((r) => setTimeout(r, 1500))
  const b = await call.state()
  expect(b.rtp.audio.totalSamplesReceived ?? 0, 'remote audio samples').toBeGreaterThan(a.rtp.audio.totalSamplesReceived ?? 0)
  const lv = await call.levels()
  expect(lv.remote, 'remote audio RMS (not silence)').toBeGreaterThan(1e-5)
  return lv
}

// Denoise on: either the SDK reports the kernel active (and the tone + noise
// input is strongly attenuated end to end), or it reports the fallback
// ('audio' error) and the original audio passes through.
//
// The attenuation is checked on pipeline.stream (what the SDK outputs) and on
// what pc2 received. Playwright's WebKit (WPE, Linux) sends the UPSTREAM audio
// of a WebAudio-processed track over WebRTC, SDK or not (see README, "WebKit
// audio"), so there the remote check is recorded, not asserted, until it's
// verified on real Safari.
async function expectDenoise(call, testInfo) {
  await expect.poll(async () => {
    const s = await call.state()
    return s.audioStats?.active || s.errors.some((e) => e.type === 'audio')
  }, { timeout: 30_000 }).toBe(true)
  const s = await call.state()
  const lv = await expectAudioReceived(call)
  if (s.audioStats?.active) {
    const remoteOk = lv.remote < RAW_TONE_RMS
    testInfo.annotations.push({ type: 'denoise', description: `${s.audioStats.model} active; RMS output ${lv.local.toFixed(4)}, remote ${lv.remote.toFixed(4)} (raw ~0.15)${remoteOk ? '' : ' -- remote NOT attenuated'}` })
    expect(lv.local, 'pipeline.stream audio: tone+noise attenuated').toBeLessThan(RAW_TONE_RMS)
    if (testInfo.project.name !== 'webkit') expect(lv.remote, 'remote audio: tone+noise attenuated').toBeLessThan(RAW_TONE_RMS)
  } else {
    const e = s.errors.find((x) => x.type === 'audio')
    testInfo.annotations.push({ type: 'denoise', description: `fallback: ${e.message}` })
    expect(e.fatal).toBe(false)
    expect(lv.remote, 'fallback passes the original audio').toBeGreaterThan(RAW_TONE_RMS)
  }
}

// ── Scenarios ───────────────────────────────────────────────────────────────

for (const backend of BACKENDS) {
  test(`background only [${backend}]`, async ({ call }) => {
    await call.start({ bg: 'color', color: '#00ff00', denoise: 'off', backend })
    const s = await call.settled()
    checkBackend(call, s, backend)
    await expectFramesFlow(call)
    await expectBackground(call)
    const st = await call.state()
    expect(st.stats.status).toBe('running')
    expect(st.errors.filter((e) => e.fatal)).toEqual([])
  })

  test(`audio only, denoise on [${backend}]`, async ({ call }, testInfo) => {
    await call.start({ bg: 'none', denoise: 'on', backend })
    const s = await call.settled()
    // Audio doesn't use the GPU: with 'auto' it must work even with no GPU.
    checkBackend(call, s, backend, { allowNone: backend === 'auto' })
    await expectFramesFlow(call)
    await expectDenoise(call, testInfo)
  })

  test(`background + audio [${backend}]`, async ({ call }, testInfo) => {
    await call.start({ bg: 'color', color: '#00ff00', denoise: 'on', backend })
    const s = await call.settled()
    checkBackend(call, s, backend)
    await expectFramesFlow(call)
    await expectBackground(call)
    await expectDenoise(call, testInfo)
  })

  test(`fail-open: unreachable weights [${backend}]`, async ({ call }, testInfo) => {
    await call.start({ bg: 'color', color: '#00ff00', denoise: 'on', backend, weights: UNREACHABLE })
    const s = await call.settled(90_000)
    if (s.ready?.backend === 'none') { call.record('none'); test.skip(true, `${NO_GPU_REASON} (weights are never fetched without a GPU)`) }
    // Startup can't finish without weights: ready rejects with a fatal error
    // (init-failed, or recovery-failed if the GPU was also lost meanwhile).
    call.record(`none (${s.readyError?.type ?? 'ready resolved?!'})`)
    expect(s.readyError?.fatal).toBe(true)
    expect(['init-failed', 'recovery-failed']).toContain(s.readyError.type)
    testInfo.annotations.push({ type: 'errors', description: s.errors.map((e) => e.type).join(', ') })
    await expectFramesFlow(call)
    await expectRawVideo(call)
    // Denoise wasm is unreachable too: audio falls back to the original.
    await expect.poll(async () => (await call.state()).errors.some((e) => e.type === 'audio'), { timeout: 20_000 }).toBe(true)
    const lv = await expectAudioReceived(call)
    expect(lv.remote).toBeGreaterThan(RAW_TONE_RMS)
  })

  test(`fail-open: GPU device lost mid-call [${backend}]`, async ({ call }) => {
    await call.start({ bg: 'color', color: '#00ff00', denoise: 'off', backend })
    const s = await call.settled()
    checkBackend(call, s, backend)
    await expectBackground(call)
    const recoveredBefore = s.recovered.length
    const lostBefore = s.errors.filter((e) => e.type === 'backend-lost').length
    await call.eval(() => window.example.pipeline._debug('lose-backend'))
    await expect.poll(async () => (await call.state()).errors.filter((e) => e.type === 'backend-lost').length).toBeGreaterThan(lostBefore)
    const lost = (await call.state()).errors.filter((e) => e.type === 'backend-lost').at(-1)
    expect(lost.fatal).toBe(false)
    // Frames keep reaching pc2 while the SDK restarts...
    await expectFramesFlow(call, 2000)
    // ...and the effect comes back (onRecovered), on the same or a new track.
    await expect.poll(async () => (await call.state()).recovered.length, { timeout: 30_000 }).toBeGreaterThan(recoveredBefore)
    const after = await call.state()
    call.record(`${checkBackendName(s)} -> ${after.recovered.at(-1).backend}${after.trackChanges ? ' (onTrackChanged -> replaceTrack)' : ''}`)
    await expectFramesFlow(call)
    await expectBackground(call)
  })
}

const checkBackendName = (s) => s.ready?.backend ?? 'none'

// Regression: no VideoFrame may be left for the garbage collector. Chromium
// reports one as "A VideoFrame was garbage collected without being closed" on
// a ~10 s timer, so a leak used to fail whichever later test's teardown that
// report happened to land in. Three pipelines on one page (create → ready →
// worker restart → destroy, then a new one), the restart alternating GPU loss
// and a killed worker; then a backend 'none' rebuild and back; then GC and wait
// past the next report.
test('no unclosed VideoFrames across restarts and destroy [auto]', async ({ call, page, browserName }) => {
  test.skip(browserName !== 'chromium', 'only Chromium reports unclosed VideoFrames')
  const gc = async () => {
    await page.evaluate(() => { for (let i = 0; i < 3; i++) globalThis.gc?.() })
    for (const w of page.workers()) await w.evaluate(() => { globalThis.gc?.() }).catch(() => {})
  }
  await call.start({ bg: 'color', color: '#00ff00', denoise: 'off', backend: 'auto' })
  for (let i = 0; i < 3; i++) {
    if (i > 0) await call.eval(() => window.example.restartPipeline())   // destroy() + a new pipeline
    const s = await call.settled()
    checkBackend(call, s, 'auto')
    // Headless Chromium's WebGPU device can still be lost (and the worker
    // restarting) just after ready: wait for a running worker first.
    await expect.poll(async () => (await call.state()).stats?.status, { timeout: 30_000 }).toBe('running')
    const before = (await call.state()).recovered.length
    await call.eval((a) => window.example.pipeline._debug(a), i % 2 ? 'terminate' : 'lose-backend')
    await expect.poll(async () => (await call.state()).recovered.length, { timeout: 30_000 }).toBeGreaterThan(before)
    await expectFramesFlow(call, 1500)
  }
  // The no-GPU rebuild: destroy() + a new pipeline with backend 'none' (main-side
  // raw output, the worker releases its input), then back to a GPU pipeline.
  for (const backend of ['none', 'auto']) {
    await call.eval((b) => { document.getElementById('controls').elements.backend.value = b; return window.example.restartPipeline() }, backend)
    const s = await call.settled()
    if (backend === 'none') expect(s.ready?.backend).toBe('none')
    await expectFramesFlow(call, 1500)
  }
  await call.eval(() => window.example.stop())
  await gc()
  await page.waitForTimeout(11_000)
  await gc()
  await page.waitForTimeout(500)
  expect(call.consoleErrors.filter((m) => /VideoFrame/.test(m)), 'unclosed VideoFrames').toEqual([])
})

test('fail-open: no GPU (backend none) [none]', async ({ call }) => {
  await call.start({ bg: 'color', color: '#00ff00', denoise: 'off', backend: 'none' })
  const s = await call.settled()
  call.record(s.ready?.backend ?? 'none')
  expect(s.ready?.backend).toBe('none')
  const e = s.errors.find((x) => x.type === 'no-backend')
  expect(e, 'no-backend error').toBeTruthy()
  expect(e.fatal).toBe(false)
  await expectFramesFlow(call)
  await expectRawVideo(call)
  const st = await call.state()
  expect(st.stats.backend).toBe('none')
  expect(st.stats.status).toBe('passthrough')
})

// Bonus: the browser's fake camera through getUserMedia (no pixel check: the
// fake pattern has no person in it).
test('getUserMedia fake camera [auto]', async ({ call, browserName }) => {
  test.skip(browserName === 'webkit', 'Playwright WebKit has no fake camera device')
  await call.start({ input: 'camera', bg: 'blur', denoise: 'off', backend: 'auto' })
  const s = await call.settled()
  checkBackend(call, s, 'auto')
  await expectFramesFlow(call)
})
