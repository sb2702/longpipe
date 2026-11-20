// @longpipe/livekit in a real LiveKit room: the example publishes the test clip
// through the Longpipe TrackProcessor (alice) and subscribes to it (bob) via a
// local livekit-server. Checks are on what bob RECEIVED.
import { test, expect } from '@playwright/test'

const GREEN = [0, 255, 0]
const CORNERS = { tl: [0.04, 0.06], tr: [0.96, 0.06], bl: [0.04, 0.94], br: [0.96, 0.94] }
const SUBJECT = { face: [0.48, 0.35], chest: [0.5, 0.82] }
const dist = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))

async function join(page, params) {
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))
  const q = new URLSearchParams({ input: 'clip', processor: 'longpipe', bg: 'color', denoise: 'off', preset: 'fast', weights: 'local', denoiseModel: 'rnnoise', ...params })
  await page.goto(`/?${q}`)
  await page.click('#join')
  await expect.poll(() => page.evaluate(() => window.example.state.running), { message: 'joined' }).toBe(true)
  return errors
}

const sample = (page, pts) => page.evaluate((p) => window.example.sampleRemote(p), pts)

// Software GPUs (CI, headless) run the effect at a few fps: wait up to 10 s for 3 new frames.
async function remoteFramesRising(page) {
  const frames = () => page.evaluate(() => window.example.remoteFrames())
  const start = await frames()
  const t0 = Date.now()
  await expect.poll(async () => (await frames()) - start, { message: 'remote frames decoded', timeout: 10_000, intervals: [500] }).toBeGreaterThanOrEqual(3)
  await page.waitForTimeout(2000)
  return Math.round(((await frames()) - start) / ((Date.now() - t0) / 1000))
}

async function expectGreenBackground(page) {
  await expect.poll(async () => {
    const s = await sample(page, { ...CORNERS, ...SUBJECT })
    if (!s) return 'no remote frame yet'
    const bad = Object.keys(CORNERS).filter((k) => dist(s.patches[k], GREEN) > 60)
    const subj = Object.keys(SUBJECT).filter((k) => dist(s.patches[k], GREEN) < 100)
    return bad.length || subj.length ? `corners off: ${bad} / subject replaced: ${subj} ${JSON.stringify(s.patches)}` : 'ok'
  }, { timeout: 60_000, intervals: [500] }).toBe('ok')
}

test.afterEach(async ({ page }) => {
  await page.click('#leave').catch(() => {})
})

test('Longpipe processor: effect reaches the subscriber, survives a source swap, denoise runs', async ({ page }, info) => {
  const errors = await join(page, { denoise: 'on' })
  await expectGreenBackground(page)
  const fps = await remoteFramesRising(page)
  const before = await page.evaluate(() => window.example.state.stats)
  console.log(`[${info.project.name}] backend ${before?.backend}, model ${before?.preset}, effect ${before?.fps} fps, remote ~${fps} fps`)

  // Camera switch: LiveKit calls processor.restart(); Longpipe keeps the same output track.
  const trackBefore = await page.evaluate(() => window.example.processedTrackId())
  await page.evaluate(() => window.example.replaceSource())
  await expectGreenBackground(page)
  await remoteFramesRising(page)
  expect(await page.evaluate(() => window.example.processedTrackId()), 'same processed track after restart').toBe(trackBefore)

  await expect.poll(() => page.evaluate(() => window.example.state.denoise?.active ?? null), { message: 'denoise active' }).toBe(true)
  expect(await page.evaluate(() => window.example.state.errors.filter((e) => e.fatal))).toEqual([])
  expect(errors).toEqual([])
})

test('camera mute/unmute: LiveKit stops the camera, the processor rebuilds on unmute', async ({ page }) => {
  // The browser's fake camera (no person in it), so this checks frames, not the effect.
  const errors = await join(page, { input: 'camera', bg: 'blur', res: '360' })
  await expect.poll(() => page.evaluate(() => window.example.state.ready?.backend ?? null)).not.toBeNull()
  await remoteFramesRising(page)
  await page.evaluate(() => window.example.muteCycle())
  await expect.poll(() => page.evaluate(async () => (await window.example.processorStats())?.status), { message: 'pipeline running again' }).toBe('running')
  await remoteFramesRising(page)
  // camera-ended is expected on mute and filtered by the processor; headless Chromium also reports backend-lost (WebGPU).
  const types = await page.evaluate(() => window.example.state.errors.map((e) => `${e.type}${e.fatal ? ' (fatal)' : ''}`))
  expect(types.filter((t) => t !== 'backend-lost')).toEqual([])
  expect(errors).toEqual([])
})

test("fail-open: backend 'none' still delivers raw video", async ({ page }) => {
  const errors = await join(page, { backend: 'none' })
  await expect.poll(() => page.evaluate(() => window.example.state.ready?.backend ?? null)).toBe('none')
  await remoteFramesRising(page)
  const s = await sample(page, CORNERS)
  const green = Object.keys(CORNERS).filter((k) => dist(s.patches[k], GREEN) < 60)
  expect(green, `corners must be the raw clip: ${JSON.stringify(s.patches)}`).toEqual([])
  expect(await page.evaluate(() => window.example.state.errors.map((e) => e.type))).toContain('no-backend')
  expect(errors).toEqual([])
})
