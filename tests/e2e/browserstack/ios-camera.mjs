// Camera-restart soak on a REAL iPhone camera (BrowserStack Automate, WebDriver).
//
// Playwright can drive iOS Safari on BrowserStack but can't answer Safari's
// native "Allow camera" prompt, so this script talks plain W3C WebDriver/Appium
// to the BrowserStack hub (no extra dependencies): it taps "Allow" in the
// NATIVE_APP context whenever the page is waiting for getUserMedia, and runs
// the example's soak hook (examples/vanilla-webrtc/soak.js) with input=camera.
//
//   export BROWSERSTACK_USERNAME=… BROWSERSTACK_ACCESS_KEY=…
//   npm run test:e2e:bs:ios-camera                     # iPhone Air, iOS 26
//   npm run test:e2e:bs:ios-camera -- --device "iPhone 17" --os 26 --cycles 10
//   ... -- --input clip                                # the test clip instead of the camera
//
// Starts the example server (HTTPS) and BrowserStack Local itself unless
// E2E_BS_BASE_URL / BROWSERSTACK_LOCAL_RUNNING=1 say they're already up.
// Writes tests/e2e/results-bs/ios-camera-<device>.json (+ screenshots) and
// exits non-zero if the soak finds a sideways/upside-down output or an fps drop.
import { spawn } from 'node:child_process'
import https from 'node:https'
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import browserstack from 'browserstack-local'

const { values: opt } = parseArgs({ options: {
  device: { type: 'string', default: 'iPhone Air' },
  os: { type: 'string', default: '26' },
  input: { type: 'string', default: 'camera' },
  camera: { type: 'string', default: 'front' },   // front | back
  cycles: { type: 'string', default: '10' },
  bg: { type: 'string', default: '5' },
  port: { type: 'string', default: process.env.E2E_PORT ?? '5182' },
} })
const USER = process.env.BROWSERSTACK_USERNAME, KEY = process.env.BROWSERSTACK_ACCESS_KEY
if (!USER || !KEY) { console.error('Set BROWSERSTACK_USERNAME and BROWSERSTACK_ACCESS_KEY'); process.exit(2) }
const REPO = fileURLToPath(new URL('../../../', import.meta.url))
const OUT = fileURLToPath(new URL('../results-bs/', import.meta.url))
mkdirSync(OUT, { recursive: true })
const tag = `${opt.device}-${opt.camera}`.replace(/\W+/g, '-').toLowerCase()
// Phones reach the tunnel at bs-local.com.
const BASE = process.env.E2E_BS_BASE_URL ?? `https://bs-local.com:${opt.port}`
const HUB = 'https://hub-cloud.browserstack.com/wd/hub'
const AUTH = 'Basic ' + Buffer.from(`${USER}:${KEY}`).toString('base64')
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a)

// ── Local server + tunnel ───────────────────────────────────────────────────
const cleanups = []
async function startServer() {
  if (process.env.E2E_BS_BASE_URL) return
  // The built example over HTTPS (see the webServer note in ../browserstack.config.mjs).
  const p = spawn('npm', ['run', 'e2e:bs:serve'], { cwd: REPO, env: { ...process.env, E2E_PORT: opt.port }, stdio: ['ignore', 'ignore', 'inherit'], detached: true })
  cleanups.push(() => { try { process.kill(-p.pid) } catch {} })
  const up = () => new Promise((res) => https.get(`https://localhost:${opt.port}/`, { rejectUnauthorized: false }, (r) => { r.resume(); res(r.statusCode === 200) }).on('error', () => res(false)))
  for (let i = 0; i < 600; i++) {
    if (await up()) return
    await sleep(500)
  }
  throw new Error('example server did not start')
}
async function startTunnel() {
  if (process.env.BROWSERSTACK_LOCAL_RUNNING === '1') return
  const local = new browserstack.Local()
  await new Promise((res, rej) => local.start({ key: KEY, force: 'true', forceLocal: 'true', logFile: `${OUT}browserstack-local.log`, ...(process.env.BROWSERSTACK_LOCAL_IDENTIFIER ? { localIdentifier: process.env.BROWSERSTACK_LOCAL_IDENTIFIER } : {}) }, (e) => (e ? rej(e) : res())))
  cleanups.push(() => new Promise((r) => local.stop(r)))
}

// ── Minimal WebDriver client ────────────────────────────────────────────────
let sid = null
async function wd(method, path, body) {
  const r = await fetch(sid ? `${HUB}/session/${sid}${path}` : `${HUB}/session`, {
    method, headers: { Authorization: AUTH, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined,
  })
  const j = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${JSON.stringify(j.value ?? j).slice(0, 300)}`)
  return j.value
}
const exec = (script, args = []) => wd('POST', '/execute/sync', { script, args })
let webContext = null
async function tapAllow() {
  // Safari's camera/microphone prompt is native UI: find "Allow" in NATIVE_APP.
  const ctxs = await wd('GET', '/contexts')
  webContext ??= ctxs.find((c) => c !== 'NATIVE_APP')
  await wd('POST', '/context', { name: 'NATIVE_APP' })
  let tapped = 0
  try {
    for (const using of [['accessibility id', 'Allow'], ['xpath', "//XCUIElementTypeButton[@name='Allow']"]]) {
      const els = await wd('POST', '/elements', { using: using[0], value: using[1] }).catch(() => [])
      for (const el of els ?? []) {
        const id = Object.values(el)[0]
        await wd('POST', `/element/${id}/click`, {}).catch(() => {})
        tapped++
      }
      if (tapped) break
    }
  } finally {
    const now = await wd('GET', '/contexts')
    webContext = now.find((c) => c !== 'NATIVE_APP') ?? webContext
    await wd('POST', '/context', { name: webContext })
  }
  return tapped
}
async function screenshot(name) {
  const b64 = await wd('GET', '/screenshot').catch(() => null)
  if (b64) writeFileSync(`${OUT}ios-camera-${tag}-${name}.png`, Buffer.from(b64, 'base64'))
}
async function sessionInfo() {
  const r = await fetch(`https://api.browserstack.com/automate/sessions/${sid}.json`, { headers: { Authorization: AUTH } })
  const s = (await r.json()).automation_session ?? {}
  return { public_url: s.public_url, video_url: s.video_url, device: s.device, os_version: s.os_version }
}
async function setStatus(status, reason) {
  await exec(`browserstack_executor: ${JSON.stringify({ action: 'setSessionStatus', arguments: { status, reason: reason.slice(0, 250) } })}`).catch(() => {})
}

// ── Run ─────────────────────────────────────────────────────────────────────
let exitCode = 1
try {
  await startServer()
  await startTunnel()
  log(`session: ${opt.device} iOS ${opt.os}, input=${opt.input}`)
  const s = await wd('POST', '', { capabilities: { alwaysMatch: {
    browserName: 'safari',
    acceptInsecureCerts: true,
    'bstack:options': {
      deviceName: opt.device, osVersion: opt.os, realMobile: 'true', local: 'true',
      acceptInsecureCerts: 'true',   // the example's throwaway HTTPS cert
      ...(process.env.BROWSERSTACK_LOCAL_IDENTIFIER ? { localIdentifier: process.env.BROWSERSTACK_LOCAL_IDENTIFIER } : {}),
      userName: USER, accessKey: KEY, projectName: 'longpipe', buildName: process.env.BROWSERSTACK_BUILD_NAME ?? 'longpipe ios-camera',
      sessionName: `camera restart soak (${opt.input})`, debug: 'true',
    },
  } } })
  sid = s?.sessionId
  if (!sid) throw new Error(`no session: ${JSON.stringify(s).slice(0, 300)}`)
  const q = new URLSearchParams({ input: opt.input, camera: opt.camera, bg: 'blur', denoise: 'off', backend: 'auto', preset: 'fast', weights: 'local' })
  await wd('POST', '/url', { url: `${BASE}/?${q}` })
  await exec("document.getElementById('start').click()")

  // Until ready: answer camera prompts.
  const t0 = Date.now()
  let st
  for (;;) {
    st = await exec('const s = window.example.state; return { running: s.running, ready: s.ready, readyError: s.readyError, log: document.getElementById("log").textContent.slice(0, 400) }')
    if (st.ready || st.readyError) break
    if (Date.now() - t0 > 120_000) throw new Error(`pipeline not ready after 120 s: ${JSON.stringify(st)}`)
    if (opt.input === 'camera' && !st.running) { const n = await tapAllow(); if (n) log('tapped Allow') }
    await sleep(1500)
  }
  log('ready:', JSON.stringify(st.ready ?? st.readyError))
  await screenshot('start')
  await exec(`window.example.startSoak({ cycles: ${Number(opt.cycles)}, bgChanges: ${Number(opt.bg)} }); return 1`)

  let phase = '', since = Date.now(), soak
  for (;;) {
    soak = await exec('return JSON.parse(JSON.stringify(window.example.soak))')
    if (soak.phase !== phase) { phase = soak.phase; since = Date.now(); log('soak:', phase) }
    if (soak.done) break
    if (opt.input === 'camera' && /opening camera/.test(phase) && Date.now() - since > 2500) { if (await tapAllow()) log('tapped Allow'); since = Date.now() }
    if (Date.now() - t0 > 600_000) throw new Error(`soak stuck at ${phase}`)
    await sleep(2000)
  }
  await screenshot('end')

  // Same checks as tests/e2e/camera-restart.e2e.mjs.
  const BAD = ['sideways', 'upside-down', 'no-frame']
  const b = soak.steps[0]
  const minFps = Math.max(0.5, (b.deliveredFps ?? 0) * 0.5)
  const problems = [
    ...(!st.ready || st.ready.backend === 'none' ? [`no GPU effect: ${JSON.stringify(st.ready ?? st.readyError)}`] : []),
    ...(soak.error ? [`error: ${soak.error}`] : []),
    ...(BAD.includes(b.verdict) ? [`baseline: ${b.verdict}`] : []),
    ...soak.steps.slice(1).flatMap((r) => [
      ...(BAD.includes(r.verdict) ? [`${r.step}: ${r.verdict} (out ${r.out}, in ${r.in})`] : []),
      ...(r.deliveredFps != null && r.deliveredFps < minFps ? [`${r.step}: ${r.deliveredFps} fps shown < ${minFps.toFixed(1)}`] : []),
    ]),
  ]
  for (const r of soak.steps) log(`${r.step}: ${r.verdict} out ${r.out} in ${r.in} | shown ${r.deliveredFps} fps (input ${r.inputFps}, getStats ${r.statsFps}) | ${r.backend}/${r.status} corr ${JSON.stringify(r.corr)}`)
  await setStatus(problems.length ? 'failed' : 'passed', problems.join('; ') || 'soak ok')
  const info = await sessionInfo()
  writeFileSync(`${OUT}ios-camera-${tag}.json`, JSON.stringify({ device: opt.device, os: opt.os, input: opt.input, ready: st.ready, problems, session: info, soak }, null, 1))
  log(problems.length ? `FAIL:\n  ${problems.join('\n  ')}` : 'PASS')
  log(`session: ${info.public_url ?? sid}`)
  exitCode = problems.length ? 1 : 0
} catch (e) {
  log('ERROR', e.message)
  if (sid) { await screenshot('error'); await setStatus('failed', e.message) }
} finally {
  if (sid) await wd('DELETE', '').catch(() => {})
  for (const c of cleanups.reverse()) await c()
  process.exit(exitCode)
}
