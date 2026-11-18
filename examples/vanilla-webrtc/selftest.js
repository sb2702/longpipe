// Device self-test: one click runs the example through the checks a manual
// test would (camera + mic, solid-colour background, denoise on, model and
// backend auto), records what happened, and POSTs the result to the dev
// server (vite.config.js appends it to selftest-results.jsonl; summarise
// with `node examples/vanilla-webrtc/selftest-table.mjs`).
//
//   1 start-up      time to `ready`, ready info, transport paths, errors
//   2 environment   browser / version / OS, GPU, SDK version (after start-up, so
//                   the click's user activation goes to getUserMedia)
//   3 video         remote framesDecoded fps over 3 s, resolution, effect visible, local fps
//   4 audio         remote audio samples, denoise model + active
//   5 GPU loss      _debug('lose-backend') → onRecovered within 20 s? frames during / after
//   6 worker crash  _debug('terminate') → same, plus onTrackChanged / replaceTrack
//   7 no GPU        rebuild with backend 'none': raw frames still arrive, non-fatal no-backend
//   8 teardown      stop(): output tracks ended
//
// Each step is independent: a failure is recorded with its error and the test
// carries on. Uses the example's own controls and window.example, like the e2e tests.
import { callStats } from './call.js'
import { toRow } from './selftest-format.js'
import pkg from '../../package.json'

const COLOR = '#ff00ff'       // magenta: rarely in a room's corners, unlike green
const RGB = [255, 0, 255]
const CORNERS = { tl: [0.04, 0.06], tr: [0.96, 0.06], bl: [0.04, 0.94], br: [0.96, 0.94] }
const RECOVER_TIMEOUT = 20_000

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const dist = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
const errText = (e) => `${e?.type ? `${e.type}: ` : ''}${e?.message ?? e}`

async function waitFor(fn, timeout, interval = 250) {
  const end = performance.now() + timeout
  for (;;) {
    const v = await fn()
    if (v) return v
    if (performance.now() > end) return null
    await sleep(interval)
  }
}

function withTimeout(promise, ms, what) {
  let t
  return Promise.race([promise, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(`${what}: timed out after ${ms / 1000} s`)), ms) })])
    .finally(() => clearTimeout(t))
}

// ── 1. Environment ──────────────────────────────────────────────────────────

function parseUA(ua) {
  const ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)
  const pick = (re) => re.exec(ua)?.[1]
  let browser = 'unknown', browserVersion = ''
  for (const [name, re] of [
    ['Edge', /Edg(?:e|A|iOS)?\/([\d.]+)/], ['Opera', /OPR\/([\d.]+)/], ['Samsung Internet', /SamsungBrowser\/([\d.]+)/],
    ['Firefox', /(?:Firefox|FxiOS)\/([\d.]+)/], ['Chrome', /(?:Chrome|CriOS)\/([\d.]+)/], ['Safari', /Version\/([\d.]+).*Safari/],
  ]) {
    const v = pick(re)
    if (v) { browser = name; browserVersion = v; break }
  }
  let os = 'unknown'
  if (/iPhone|iPod/.test(ua)) os = `iOS ${(pick(/OS ([\d_]+)/) ?? '').replace(/_/g, '.')}`
  else if (/iPad/.test(ua)) os = `iPadOS ${(pick(/OS ([\d_]+)/) ?? '').replace(/_/g, '.')}`
  else if (ios) os = 'iPadOS (desktop UA)'
  else if (/Android/.test(ua)) os = `Android ${pick(/Android ([\d.]+)/) ?? ''}`.trim()
  else if (/Windows NT/.test(ua)) os = 'Windows'
  else if (/Mac OS X/.test(ua)) os = 'macOS'   // frozen at 10_15_7 in UA strings
  else if (/CrOS/.test(ua)) os = 'ChromeOS'
  else if (/Linux/.test(ua)) os = 'Linux'
  // On iOS/iPadOS every browser is WebKit; Safari's version is the WebKit release.
  if (ios && browser === 'Safari') browserVersion = pick(/Version\/([\d.]+)/) ?? browserVersion
  return { browser, browserVersion, os, ios }
}

async function gpuInfo() {
  const out = {}
  try {
    const adapter = await navigator.gpu?.requestAdapter()
    if (adapter) {
      const info = adapter.info ?? (await adapter.requestAdapterInfo?.())
      if (info) out.webgpu = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description }
      out.webgpuFallbackAdapter = adapter.isFallbackAdapter ?? info?.isFallbackAdapter ?? false
    } else {
      out.webgpu = null
    }
  } catch (e) { out.webgpuError = String(e?.message ?? e) }
  try {
    const gl = document.createElement('canvas').getContext('webgl2') ?? document.createElement('canvas').getContext('webgl')
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info')
      out.webglRenderer = gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER)
      out.webglVendor = gl.getParameter(ext ? ext.UNMASKED_VENDOR_WEBGL : gl.VENDOR)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  } catch (e) { out.webglError = String(e?.message ?? e) }
  return out
}

async function environment() {
  const ua = navigator.userAgent
  const env = { userAgent: ua, sdk: pkg.version, ...parseUA(ua) }
  // Chromium: the real (unfrozen) browser and OS versions.
  const uad = navigator.userAgentData
  if (uad?.getHighEntropyValues) {
    try {
      const h = await uad.getHighEntropyValues(['fullVersionList', 'platformVersion', 'model', 'architecture'])
      env.userAgentData = h
      const brand = h.fullVersionList?.find((b) => !/Not.?A.?Brand|Chromium/i.test(b.brand)) ?? h.fullVersionList?.find((b) => /Chromium/.test(b.brand))
      if (brand) {
        env.browser = { 'Google Chrome': 'Chrome', 'Microsoft Edge': 'Edge' }[brand.brand] ?? brand.brand
        env.browserVersion = brand.version
      }
      if (h.platform === 'macOS' && h.platformVersion) env.os = `macOS ${h.platformVersion}`
      else if (h.platform === 'Windows' && h.platformVersion) {
        // platformVersion major ≥ 13 is Windows 11.
        env.os = `Windows ${parseInt(h.platformVersion, 10) >= 13 ? '11' : '10'}`
      } else if (h.platform === 'Android' && h.platformVersion) env.os = `Android ${h.platformVersion}${h.model ? ` (${h.model})` : ''}`
    } catch { /* not allowed here: keep the UA parse */ }
  }
  if (env.ios) env.engineNote = 'iOS/iPadOS: every browser is WebKit'
  env.gpu = await gpuInfo()
  env.screen = `${screen.width}×${screen.height}@${devicePixelRatio}`
  env.hardwareConcurrency = navigator.hardwareConcurrency
  return env
}

// ── Measurements on the running call ────────────────────────────────────────

async function rtp(example) {
  return example.call ? callStats(example.call) : null
}
const decoded = async (example) => (await rtp(example))?.video.framesDecoded ?? 0

// Corners of the remote video against the background colour: how many match.
function cornersMatching(example) {
  const s = example.sampleRemote(CORNERS)
  if (!s) return null
  const match = Object.keys(CORNERS).filter((k) => dist(s.patches[k], RGB) <= 60)
  return { match: match.length, patches: s.patches }
}

const stats = async (example) => (example.pipeline ? example.pipeline.getStats().catch(() => null) : null)

// Trigger a failure and watch the recovery: onRecovered within 20 s, how long,
// backend before → after, remote frames during (trigger → recovery, at least
// 1 s) and for 3 s after, the longest stretch with no new decoded frame, restarts.
// The SDK allows 3 automatic restarts per 60 s of healthy running. If earlier
// failures (this test's, or WebGPU device losses at start-up) used up 2, the
// pipeline is rebuilt first so this step gets its own budget (recorded as rebuiltFirst).
const MAX_RESTARTS = 3
async function recoveryStep(example, action) {
  const st = example.state
  await waitFor(async () => (await stats(example))?.status === 'running', 15_000)
  let rebuiltFirst = null
  const pre = await stats(example)
  if (pre && pre.restarts >= MAX_RESTARTS - 1) {
    rebuiltFirst = `${pre.restarts} restarts already used`
    await example.restartPipeline()
    await waitFor(() => st.ready || st.readyError, 30_000)
    await waitFor(async () => (await stats(example))?.status === 'running', 15_000)
  }
  const before = await stats(example)
  const backendBefore = before?.backend ?? st.ready?.backend ?? 'none'
  const recBefore = st.recovered.length
  const trackBefore = st.trackChanges
  const replaceBefore = st.replaceTracks.length
  const errBefore = st.errors.length
  const sender = example.call?.senders.video
  const senderTrackBefore = sender?.track

  const t0 = performance.now()
  let last = await decoded(example)
  const startCount = last
  let lastChange = t0, maxStall = 0, recoverAt = null
  await example.pipeline._debug(action)
  // Poll every 100 ms until recovery (or timeout), but at least 1 s.
  while (performance.now() - t0 < RECOVER_TIMEOUT) {
    await sleep(100)
    const n = await decoded(example)
    const now = performance.now()
    if (n > last) { maxStall = Math.max(maxStall, now - lastChange); lastChange = now; last = n }
    if (recoverAt == null && st.recovered.length > recBefore) recoverAt = now
    if (recoverAt != null && now - t0 >= 1000) break
  }
  const end = performance.now()
  maxStall = Math.max(maxStall, end - lastChange)
  const framesDuring = last - startCount
  // 3 s after: frames still arriving, effect back.
  const a0 = await decoded(example)
  await sleep(3000)
  const framesAfter = (await decoded(example)) - a0
  const recovered = recoverAt != null
  let effectAfter = null
  if (recovered) effectAfter = !!(await waitFor(() => (cornersMatching(example)?.match ?? 0) >= 3, 5000))
  const after = await stats(example)
  const out = {
    recovered,
    recoverMs: recovered ? Math.round(recoverAt - t0) : null,
    recoveredInfo: st.recovered[recBefore] ?? null,
    backendBefore,
    backendAfter: st.recovered.at(-1)?.backend ?? after?.backend ?? 'none',
    framesDuring, framesAfter, maxStallMs: Math.round(maxStall),
    effectAfter,
    rebuiltFirst,
    restartsBefore: before?.restarts ?? null,
    restartsAfter: after?.restarts ?? null,
    errors: st.errors.slice(errBefore),
    trackChanged: st.trackChanges > trackBefore,
    senderTrackReplaced: !!sender && sender.track !== senderTrackBefore,
  }
  if (out.trackChanged) {
    await waitFor(() => st.replaceTracks.length > replaceBefore, 3000, 100)
    out.replaceTrack = st.replaceTracks[replaceBefore] ?? { ok: false, error: 'no result' }
  }
  out.ok = recovered && framesDuring > 0 && framesAfter > 0 && (!out.trackChanged || out.replaceTrack.ok)
  return out
}

// ── The run ─────────────────────────────────────────────────────────────────

const STEPS = [
  ['startup', 'Start-up'],
  ['env', 'Environment'],
  ['video', 'Processed video over WebRTC'],
  ['audio', 'Audio'],
  ['gpuLoss', 'GPU loss → recovery'],
  ['workerCrash', 'Worker crash → recovery'],
  ['noGpu', 'No-GPU fallback'],
  ['teardown', 'Teardown'],
]

export async function runSelfTest({ example, form, onProgress = () => {} }) {
  // ?label=… tags a run (e.g. label=headless for automated runs, so the table can leave them out).
  const label = new URLSearchParams(location.search).get('label')
  const result = { kind: 'longpipe-selftest', v: 1, startedAt: new Date().toISOString(), page: location.pathname + location.search, ...(label ? { label } : {}), steps: {} }
  const t0 = performance.now()
  const st = example.state

  const run = async (key, fn, timeout) => {
    const label = STEPS.find(([k]) => k === key)[1]
    onProgress({ key, label, phase: 'run', index: STEPS.findIndex(([k]) => k === key), total: STEPS.length })
    const s0 = performance.now()
    let out
    try {
      out = await withTimeout(fn(), timeout, label)
      out.ok ??= true
    } catch (e) {
      out = { ok: false, error: errText(e) }
    }
    out.ms = Math.round(performance.now() - s0)
    result.steps[key] = out
    onProgress({ key, label, phase: 'done', step: out, index: STEPS.findIndex(([k]) => k === key) + 1, total: STEPS.length })
    return out
  }

  await run('startup', async () => {
    if (st.running) await example.stop()
    const f = form.elements
    f.input.value = 'camera'
    f.bg.value = 'color'
    f.color.value = COLOR
    f.denoise.value = 'on'
    f.backend.value = 'auto'
    f.model.value = 'auto'
    f.bypass.checked = false
    f.flash.checked = false
    await example.start()
    await waitFor(() => st.ready || st.readyError, 30_000)
    const s = await stats(example)
    const out = {
      readyMs: st.readyMs,
      ready: st.ready,
      readyError: st.readyError,
      paths: s ? `${s.inputPath} → ${s.outputPath}` : null,
      restarts: s?.restarts ?? null,   // e.g. WebGPU device lost during start-up
      errors: [...st.errors],
    }
    if (!st.ready && !st.readyError) throw new Error('ready never settled in 30 s')
    out.ok = !!st.ready && st.ready.backend !== 'none'
    if (st.readyError) out.error = errText(st.readyError)
    else if (st.ready.backend === 'none') out.error = 'no GPU backend (fell back to none)'
    return out
  }, 60_000)

  // After start-up, so the click's user activation goes to getUserMedia (as with Start).
  await run('env', async () => {
    result.env = await environment()
    return { sdk: result.env.sdk }
  }, 10_000)

  await run('video', async () => {
    if (!example.call) throw new Error('no call')
    // Effect visible: corners of the remote video match the background colour.
    const effect = await waitFor(() => { const c = cornersMatching(example); return c && c.match >= 3 ? c : null }, 10_000)
    const a = await rtp(example)
    const t = performance.now()
    await sleep(3000)
    const b = await rtp(example)
    const delta = (b.video.framesDecoded ?? 0) - (a.video.framesDecoded ?? 0)
    const s = await stats(example)
    const localFrames = HTMLVideoElement.prototype.requestVideoFrameCallback ? await example.countLocalFrames(1000) : null
    const last = effect ?? cornersMatching(example)
    return {
      ok: delta > 0 && !!effect,
      framesDecodedDelta: delta,
      remoteFps: Math.round(delta / ((performance.now() - t) / 1000)),
      width: b.video.frameWidth, height: b.video.frameHeight,
      effect: !!effect,
      cornersMatching: last?.match ?? null, corners: last?.patches ?? null,
      localOutputFps: s?.fps ?? null,
      localRenderedFps: localFrames,
      inputFps: s?.inputFps == null ? null : Math.round(s.inputFps),
      latencyMs: s?.latencyMs ?? null,
    }
  }, 25_000)

  await run('audio', async () => {
    if (!example.call) throw new Error('no call')
    // Denoise reports active, or the SDK says it fell back.
    await waitFor(() => example.pipeline?.getAudioStats()?.active || st.errors.some((e) => e.type === 'audio'), 10_000)
    const a = await rtp(example)
    await sleep(1500)
    const b = await rtp(example)
    const samples = (b.audio.totalSamplesReceived ?? 0) - (a.audio.totalSamplesReceived ?? 0)
    const as = example.pipeline?.getAudioStats() ?? null
    const audioErr = st.errors.find((e) => e.type === 'audio' || e.type === 'audio-suspended')
    let remoteRms = null
    try { remoteRms = (await withTimeout(example.audioLevels(), 3000, 'audio level')).remote } catch { /* optional */ }
    return {
      ok: samples > 0 && !!as?.active,
      remoteSamplesDelta: samples,
      remotePacketsReceived: b.audio.packetsReceived ?? null,
      remoteRms,
      denoiseModel: as?.model ?? null,
      denoiseActive: as?.active ?? false,
      sampleRate: as?.sampleRate ?? null,
      contextState: example.pipeline?.denoiser?.state ?? null,   // internal; 'running' expected
      audioTrackChanges: [...(st.audioTrackChanges ?? [])],
      ...(audioErr ? { error: errText(audioErr) } : {}),
    }
  }, 20_000)

  await run('gpuLoss', () => {
    if (!example.pipeline) throw new Error('no pipeline')
    return recoveryStep(example, 'lose-backend')
  }, 80_000)

  await run('workerCrash', () => {
    if (!example.pipeline) throw new Error('no pipeline')
    return recoveryStep(example, 'terminate')
  }, 80_000)

  await run('noGpu', async () => {
    if (!example.call) throw new Error('no call')
    const errBefore = st.errors.length
    form.elements.backend.value = 'none'
    try {
      await example.restartPipeline()
      await waitFor(() => st.ready || st.readyError, 15_000)
      const e = st.errors.slice(errBefore).find((x) => x.type === 'no-backend')
      const d0 = await decoded(example)
      await sleep(2000)
      const framesDelta = (await decoded(example)) - d0
      const c = cornersMatching(example)
      const s = await stats(example)
      return {
        ok: st.ready?.backend === 'none' && framesDelta > 0 && !!e && !e.fatal,
        readyBackend: st.ready?.backend ?? null,
        status: s?.status ?? null,
        framesDelta,
        noBackendError: !!e, noBackendFatal: e?.fatal ?? null,
        // Raw video: the corners shouldn't be the background colour any more.
        rawVideo: c ? c.match < 3 : null,
      }
    } finally {
      form.elements.backend.value = 'auto'
    }
  }, 30_000)

  await run('teardown', async () => {
    const tracks = example.pipeline?.stream.getTracks() ?? []
    await example.stop()
    await sleep(500)
    const states = tracks.map((t) => `${t.kind}:${t.readyState}`)
    return { ok: !st.running && tracks.every((t) => t.readyState === 'ended'), tracks: states }
  }, 10_000)

  result.durationMs = Math.round(performance.now() - t0)
  result.ok = Object.values(result.steps).every((s) => s.ok)
  result.row = toRow(result)
  return result
}

// ── UI ──────────────────────────────────────────────────────────────────────

export function setupSelfTest({ example, form, log, show }) {
  const $ = (id) => document.getElementById(id)
  const section = $('selftest')
  if (show) { section.hidden = false; section.classList.add('prominent') }
  let last = null
  let running = false

  const summary = (s) => {
    if (s.error) return s.error
    const bits = []
    if (s.readyMs != null) bits.push(`ready in ${(s.readyMs / 1000).toFixed(1)} s`)
    if (s.ready) bits.push(`${s.ready.backend} ${s.ready.preset ?? ''} ${s.ready.dtype ?? ''}`.trim())
    if (s.paths) bits.push(s.paths)
    if (s.errors?.length) bits.push(`errors: ${s.errors.map((e) => e.type).join(', ')}`)
    if (s.rebuiltFirst) bits.push(`(pipeline rebuilt first: ${s.rebuiltFirst})`)
    if (s.remoteFps != null) bits.push(`${s.remoteFps} fps ${s.width}×${s.height}, effect ${s.effect ? 'visible' : 'NOT visible'}, local ${s.localOutputFps} fps`)
    if (s.remoteSamplesDelta != null) bits.push(`${s.remoteSamplesDelta} samples, ${s.denoiseModel ?? 'no denoise'} ${s.denoiseActive ? 'active' : 'inactive'}`)
    if (s.recovered != null) bits.push(s.recovered ? `recovered in ${(s.recoverMs / 1000).toFixed(1)} s, ${s.backendBefore} → ${s.backendAfter}, frames ${s.framesDuring} during / ${s.framesAfter} after${s.trackChanged ? `, track replaced (${s.replaceTrack?.ok ? 'ok' : 'failed'})` : ''}` : 'no onRecovered in 20 s')
    if (s.framesDelta != null) bits.push(`${s.framesDelta} raw frames, no-backend ${s.noBackendError ? (s.noBackendFatal ? 'fatal' : 'non-fatal') : 'missing'}`)
    if (s.tracks) bits.push(s.tracks.join(', ') || 'no tracks')
    if (s.sdk) bits.push(`SDK ${s.sdk}`)
    return bits.join(' · ')
  }

  async function go() {
    if (running) return
    running = true
    section.hidden = false
    $('st-run').disabled = $('selftest-open').disabled = true
    $('st-steps').innerHTML = ''
    $('st-result').hidden = true
    $('st-saved').textContent = ''
    $('st-progress').hidden = false
    $('st-status').textContent = 'running…'
    const items = {}
    try {
      last = await runSelfTest({
        example, form,
        onProgress: ({ key, label, phase, step, index, total }) => {
          $('st-progress').value = index / total
          let li = items[key]
          if (!li) { li = items[key] = document.createElement('li'); $('st-steps').append(li) }
          li.className = phase === 'run' ? 'run' : step.ok ? 'ok' : 'fail'
          li.innerHTML = ''
          li.append(label + (phase === 'run' ? '…' : ': '))
          if (phase === 'done') {
            const d = document.createElement('span')
            d.className = 'detail'
            d.textContent = summary(step)
            li.append(d)
          }
        },
      })
      window.example.selftest = last
      $('st-status').textContent = `${last.ok ? 'all passed' : 'some checks failed'} (${(last.durationMs / 1000).toFixed(0)} s)`
      $('st-row').value = last.row
      $('st-result').hidden = false
      log(`self-test done: ${last.row}`)
      await save(last)
    } catch (e) {
      $('st-status').textContent = `self-test crashed: ${e?.message ?? e}`
    } finally {
      running = false
      $('st-run').disabled = $('selftest-open').disabled = false
      $('st-progress').hidden = true
    }
  }

  async function save(result) {
    const el = $('st-saved')
    el.className = ''
    el.textContent = 'saving…'
    try {
      const res = await fetch('/__selftest', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(result) })
      if (!res.ok) throw new Error(`HTTP ${res.status} ${await res.text().catch(() => '')}`.trim())
      el.textContent = 'saved ✓'
      el.className = 'good'
      result.saved = true
    } catch (e) {
      el.textContent = `not saved: ${e?.message ?? e} (use Copy JSON)`
      el.className = 'bad'
      result.saved = false
    }
  }

  $('st-run').addEventListener('click', go)
  $('selftest-open').addEventListener('click', go)
  $('st-copy').addEventListener('click', async () => {
    if (!last) return
    const text = JSON.stringify(last, null, 2)
    try {
      await navigator.clipboard.writeText(text)
      $('st-copy').textContent = 'Copied ✓'
    } catch {
      // No clipboard permission (or not a secure context): put it in the textarea to copy by hand.
      $('st-row').value = text
      $('st-row').rows = 12
      $('st-row').select()
      $('st-copy').textContent = 'Select-all + copy the box'
    }
  })
}
