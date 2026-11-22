// Camera restart soak (from a user report on iOS Safari: after camera off/on via
// setInputStream(), app switches and background changes, the output turned 90°
// sideways and slowed down). Cycles placeholder → new camera ×10 and the
// background ×5; after each step checks the output's orientation (aspect +
// which rotation of the upright input its content matches) and the fps the page
// actually presents (requestVideoFrameCallback), via the example's soak hook
// (examples/vanilla-webrtc/soak.js).
//
// Input: the browser's fake camera where there is one (Chromium, Firefox);
// elsewhere (BrowserStack devices: the camera prompt can't be answered from
// Playwright) a fresh test-clip canvas track per "camera on". That exercises
// setInputStream() and the probe, but not real camera rotation metadata: for
// the real iPhone camera see tests/e2e/browserstack/ios-camera.mjs.
import { test, expect } from './fixtures.mjs'

const BAD = ['sideways', 'upside-down', 'no-frame']

test('camera restart soak: setInputStream x10, background x5 [auto]', async ({ page, browserName }, testInfo) => {
  test.setTimeout(300_000)
  const bs = testInfo.project.metadata?.browserstack
  const fakeCam = !bs && browserName !== 'webkit'
  const input = process.env.E2E_SOAK_INPUT ?? (fakeCam ? 'camera' : 'clip')
  testInfo.annotations.push({ type: 'input', description: input === 'camera' ? 'fake camera (getUserMedia)' : 'test clip (new canvas track per camera-on)' })
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))

  const q = new URLSearchParams({ input, bg: 'blur', denoise: 'off', backend: 'auto', preset: 'fast', weights: 'local', denoiseModel: 'rnnoise' })
  await page.goto(`/?${q}`)
  await page.click('#start')
  await expect.poll(() => page.evaluate(() => !!(window.example.state.ready || window.example.state.readyError)), { timeout: 90_000 }).toBe(true)
  const ready = await page.evaluate(() => JSON.parse(JSON.stringify({ ready: window.example.state.ready, err: window.example.state.readyError })))
  testInfo.annotations.push({ type: 'backend', description: ready.ready?.backend ?? `none (${ready.err?.type})` })
  test.skip(!ready.ready || ready.ready.backend === 'none', `no GPU effect here (${JSON.stringify(ready)}); the soak needs a running effect`)

  await page.evaluate(() => window.example.startSoak({ cycles: 10, bgChanges: 5 }))
  let last = ''
  await expect.poll(async () => {
    // A remote device's evaluate can fail transiently (iOS): just poll again.
    const s = await page.evaluate(() => ({ phase: window.example.soak.phase, done: window.example.soak.done, n: window.example.soak.steps.length })).catch(() => ({ phase: last, done: false }))
    if (s.phase !== last) { last = s.phase; console.log(`[${testInfo.project.name}] soak: ${s.phase} (${s.n} steps)`) }
    return s.done
  }, { timeout: 240_000, intervals: [2000] }).toBe(true)
  // As a string: iOS's remote evaluate chokes on returning the nested object.
  const soak = JSON.parse(await page.evaluate(() => JSON.stringify(window.example.soak))
    .catch(() => page.evaluate(() => JSON.stringify(window.example.soak))))

  // The numbers, for the report and the summary.
  const rows = soak.steps.map((r) => `${r.step}: ${r.verdict} out ${r.out} in ${r.in} | shown ${r.deliveredFps} fps (input ${r.inputFps}, getStats ${r.statsFps}) | ${r.backend}/${r.status} corr ${JSON.stringify(r.corr)}`)
  console.log(`[${testInfo.project.name}] camera restart soak\n  ${rows.join('\n  ')}`)
  await testInfo.attach('soak.json', { body: JSON.stringify(soak, null, 1), contentType: 'application/json' })
  for (const r of rows) testInfo.annotations.push({ type: 'step', description: r })

  expect(soak.error, 'soak error').toBeNull()
  const base = soak.steps[0]
  expect(BAD, `baseline orientation (${JSON.stringify(base)})`).not.toContain(base.verdict)
  // Software GPUs show ~2-3 fps at 720p; what matters is a drop against this run's baseline.
  const minFps = Math.max(0.5, (base.deliveredFps ?? 0) * 0.5)
  const problems = soak.steps.slice(1).flatMap((r) => [
    ...(BAD.includes(r.verdict) ? [`${r.step}: ${r.verdict} (out ${r.out}, in ${r.in})`] : []),
    ...(r.deliveredFps != null && r.deliveredFps < minFps ? [`${r.step}: ${r.deliveredFps} fps shown < ${minFps.toFixed(1)} (baseline ${base.deliveredFps})`] : []),
    ...(r.status && r.status !== 'running' ? [`${r.step}: status ${r.status}`] : []),
  ])
  expect(problems, 'orientation / delivered-fps regressions').toEqual([])
  expect(errors, 'page errors').toEqual([])
  await page.evaluate(() => window.example.stop())
})
