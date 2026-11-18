// Longpipe in a plain WebRTC call:
//
//   getUserMedia → EffectsPipeline → pipeline.stream → pc1.addTrack → pc2 → <video>
//
// Everything Longpipe-specific is in startPipeline() and the three callbacks
// (onError, onRecovered, onTrackChanged). The rest is ordinary WebRTC + UI.
//
// URL parameters preset the controls (handy on a phone, and used by the e2e
// tests): ?input=clip&bg=color&color=%2300ff00&denoise=on&backend=webgl2
//   &model=auto|xs|small|medium|large|xl|mediapipe&bypass=1&flash=1
// plus some that have no control:
//   preset=fast|balanced|quality  older spelling of model= (xs | medium | xl)
//   weights=local|<url> where weights come from ('local' = ./assets/models/,
//                       see `npm run e2e:assets`; default: Longpipe's CDN)
//   denoiseModel=rnnoise|dfn|…  (default 'auto')
//   camera=front|back   which phone camera
//   debug=1             verbose [longpipe/…] console logging
import { EffectsPipeline, PRESETS } from 'longpipe'
import { startLoopbackCall, callStats, latencyBreakdown } from './call.js'
import { cameraInput, clipInput, generatedBackgroundImage, withFlashMarker } from './inputs.js'

const $ = (id) => document.getElementById(id)
const form = $('controls')
const params = new URLSearchParams(location.search)
for (const name of ['input', 'bg', 'color', 'denoise', 'backend', 'model']) {
  if (params.has(name)) form.elements[name].value = params.get(name)
}
// ?preset=fast|balanced|quality (what the e2e tests pass) → the matching tier.
const PRESET_PARAM = { fast: 'xs', balanced: 'medium', quality: 'xl', auto: 'auto' }[params.get('preset')]
if (PRESET_PARAM && !params.has('model')) form.elements.model.value = PRESET_PARAM
form.elements.bypass.checked = params.get('bypass') === '1'
form.elements.flash.checked = params.get('flash') === '1'
// MP4 (H.264) first; the WebM (VP9) copy that `npm run e2e:assets` makes with ffmpeg covers
// Firefox builds that can't decode H.264 (e.g. Playwright's Firefox on CI runners).
const CLIP_URL = ['./assets/media/woman_on_video_call_360p.mp4', './assets/media/woman_on_video_call_360p.webm']
// ?camera=back|front picks the phone camera (front/back rotation + mirroring checks).
const CAMERA_FACING = { back: 'environment', front: 'user' }[params.get('camera')]
const weightsParam = params.get('weights')
const WEIGHTS = weightsParam === 'local' ? new URL('./assets/models/', location.href).href : weightsParam || undefined

// What the page knows, in one place (the e2e tests read it via window.example).
const state = {
  running: false,
  config: null,
  ready: null,          // ReadyInfo once `pipeline.ready` resolves
  readyError: null,     // { type, fatal, message } if it rejected
  errors: [],           // every onError, in order
  recovered: [],        // every onRecovered
  trackChanges: 0,      // onTrackChanged count
  audioTrackChanges: [], // onAudioTrackChanged calls ({ denoised })
  stats: null,          // last pipeline.getStats() (null while bypassed)
  audioStats: null,     // last pipeline.getAudioStats()
  rtp: null,            // last callStats()
  remoteFps: 0,
  bypass: false,        // raw camera + mic on the senders, no pipeline
  model: null,          // the model control value the current pipeline runs ('auto', 'xs'…, 'mediapipe')
  livePreset: null,     // last model name the worker reported (getStats().preset)
  latency: null,        // { captureMs, longpipeMs, encodeMs, jitterMs, decodeMs, totalMs } (ms, last ~1 s)
}

let input = null       // { stream, stop }
let pipeline = null
let call = null
let statsTimer = null

function log(msg) {
  const line = `${new Date().toISOString().slice(11, 19)} ${msg}`
  $('log').textContent = line + '\n' + $('log').textContent
  console.log('[example]', msg)
}

function readConfig() {
  const f = form.elements
  return {
    input: f.input.value,
    bg: f.bg.value,
    color: f.color.value,
    denoise: f.denoise.value === 'on',
    backend: f.backend.value,
    model: f.model.value,
    bypass: f.bypass.checked,
    flash: f.flash.checked,
    denoiseModel: params.get('denoiseModel') ?? 'auto',
  }
}

// The model control → EffectsPipeline options.
//   'auto'             benchmark the device, then the adaptive controller moves between tiers
//   'xs' … 'xl'        one fixed Longpipe tier (a PRESETS entry), no adaptive controller
//   'mediapipe'        Google's MediaPipe Selfie Segmenter (background effects only)
const isTier = (m) => PRESETS.some((p) => p.model === m)
function modelOptions(model) {
  if (model === 'mediapipe') return { model: 'mediapipe-selfie' }
  return { model: 'longpipe', preset: model === 'auto' ? 'auto' : PRESETS.find((p) => p.model === model) }
}

async function backgroundFor(cfg) {
  switch (cfg.bg) {
    case 'blur': return 'blur'
    case 'color': return { color: cfg.color }
    case 'image': return generatedBackgroundImage()   // any URL / <img> / ImageBitmap works
    default: return 'none'
  }
}

// ── Longpipe ────────────────────────────────────────────────────────────────

// Not async on purpose: an EffectsPipeline is a thenable (`await pipeline`
// waits for `ready`), so returning one from an async function, or awaiting it,
// blocks until the effect is live and throws if startup failed. Keep the
// instance in a variable and use `pipeline.ready` when you want to wait.
function startPipeline(cfg, stream, background) {
  const p = new EffectsPipeline(stream, {
    background,
    backend: cfg.backend,                 // 'auto' in production
    ...modelOptions(cfg.model),
    weightsBaseUrl: WEIGHTS,
    debug: params.has('debug'),
    // Denoise is built in once; the toggle switches it live (setDenoise).
    audio: { denoise: { model: cfg.denoiseModel, enabled: cfg.denoise } },

    // Every failure, typed. Non-fatal = the SDK degraded (raw video, original
    // audio, previous background…) and is handling it. Fatal = effects won't
    // come back without a new pipeline. The output track keeps flowing either way.
    onError: (e) => {
      state.errors.push({ type: e.type, fatal: e.fatal, message: e.message })
      $('lastError').textContent = `${e.type}${e.fatal ? ' (fatal)' : ''}: ${e.message}`
      $('lastError').className = ''
      log(`onError ${e.type} fatal=${e.fatal}: ${e.message}`)
    },
    onRecovered: (info) => {
      state.recovered.push(info)
      log(`onRecovered ${info.reason} → ${info.backend} after ${info.attempts} attempt(s)`)
    },
    // Safari only, after its worker died: the output video track was replaced.
    // The canonical fix is one line, no renegotiation:
    onTrackChanged: ({ videoTrack, reason }) => {
      state.trackChanges++
      call?.senders.video?.replaceTrack(videoTrack)
      $('local').srcObject = p.stream
      log(`onTrackChanged (${reason}): replaced the sender's video track`)
    },
    // Denoise only, and only if the AudioContext couldn't start (no user
    // gesture left, see unlockAudio() in start()): the output audio moves to
    // the original mic audio, then back to the denoised track once it runs.
    onAudioTrackChanged: ({ audioTrack, denoised }) => {
      state.audioTrackChanges.push({ denoised })
      call?.senders.audio?.replaceTrack(audioTrack)?.catch(() => {})
      log(`onAudioTrackChanged: ${denoised ? 'denoised' : 'original audio (AudioContext not running)'}`)
    },
  })
  // `pipeline.stream` is usable right away (raw camera until the effect is
  // live); `ready` says when it is, and which backend it got.
  p.ready.then(
    (info) => { if (pipeline === p) { state.ready = info; log(`ready: ${JSON.stringify(info)}`) } },
    (e) => { if (pipeline === p) { state.readyError = { type: e.type, fatal: e.fatal, message: e.message }; log(`ready rejected: ${e.type}`) } },
  )
  return p
}

// ── Call ────────────────────────────────────────────────────────────────────

async function start() {
  if (state.running) return
  // Still inside the click: let the denoise AudioContext start now. By the time
  // getUserMedia resolves (and the pipeline exists) the gesture has expired.
  EffectsPipeline.unlockAudio()
  const cfg = readConfig()
  Object.assign(state, { running: true, config: cfg, ready: null, readyError: null, errors: [], recovered: [], trackChanges: 0, audioTrackChanges: [], bypass: cfg.bypass, model: null, livePreset: null, latency: null })
  $('start').disabled = true
  try {
    input = cfg.input === 'clip' ? await clipInput(CLIP_URL) : await cameraInput({ facingMode: CAMERA_FACING })
    if (cfg.flash) input = await withFlashMarker(input, { onFlash: (on) => { $('flash').className = on ? 'on' : '' } })
    $('flash').hidden = !cfg.flash
    let outgoing = input.stream
    if (!cfg.bypass) {
      pipeline = startPipeline(cfg, input.stream, await backgroundFor(cfg))
      state.model = cfg.model
      outgoing = pipeline.stream
    }
    $('local').srcObject = outgoing
    call = await startLoopbackCall(outgoing, {
      onRemoteTrack: (_track, remote) => { $('remote').srcObject = remote },
    })
    $('stop').disabled = false
    statsTimer = setInterval(() => updateStats(true), 1000)
    log(`started: ${JSON.stringify(cfg)}`)
  } catch (e) {
    log(`start failed: ${e?.message ?? e}`)
    $('lastError').textContent = String(e?.message ?? e)
    $('lastError').className = ''
    await stop()
    throw e
  }
}

// Put `stream`'s tracks on the existing senders: no renegotiation.
async function sendStream(stream) {
  await call.senders.video?.replaceTrack(stream.getVideoTracks()[0] ?? null)
  await call.senders.audio?.replaceTrack(stream.getAudioTracks()[0] ?? null)
  $('local').srcObject = stream
}

// Backend and model family are construction options, so changing them means a
// new pipeline. The call stays up: destroy the old pipeline, build the new one
// on the same input, swap the tracks on the existing senders.
async function restartPipeline() {
  if (!state.running || state.bypass) return
  const cfg = readConfig()
  Object.assign(state, { config: cfg, ready: null, readyError: null, stats: null, livePreset: null })
  pipeline?.destroy()
  pipeline = startPipeline(cfg, input.stream, await backgroundFor(cfg))
  state.model = cfg.model
  await sendStream(pipeline.stream)
  log(`pipeline restarted: backend=${cfg.backend} model=${cfg.model}`)
}

// Model control. Between Longpipe tiers the switch is live (setPreset: same
// pipeline, same track; it also stops the adaptive controller). 'auto' and
// MediaPipe are construction options, so those rebuild.
async function switchModel() {
  if (!state.running || state.bypass) return
  const model = form.elements.model.value
  if (isTier(model) && state.model !== 'mediapipe' && pipeline) {
    state.config = readConfig()
    const p = pipeline
    await p.setPreset(PRESETS.find((x) => x.model === model))
    if (p === pipeline) state.model = model
    log(`setPreset(${model}): live, same pipeline`)
  } else {
    await restartPipeline()
  }
}

// Bypass: the raw camera + mic go straight to the senders and the pipeline is
// destroyed (nothing competing for the CPU/GPU), which gives the WebRTC-only
// latency baseline. Off: a new pipeline, tracks swapped back. No renegotiation
// either way.
async function setBypass(on) {
  if (!state.running || on === state.bypass) return
  state.bypass = on
  state.config = readConfig()
  if (on) {
    const old = pipeline
    pipeline = null
    Object.assign(state, { stats: null, audioStats: null, model: null })
    await sendStream(input.stream)
    old?.destroy()
    log('bypass on: raw camera + mic on the senders, pipeline destroyed')
  } else {
    await restartPipeline()
    log('bypass off: pipeline back on the senders')
  }
}

// Control changes run one at a time (a rebuild awaits a background load).
let queue = Promise.resolve()
const serial = (what, fn) => {
  queue = queue.then(fn).catch((e) => log(`${what} failed: ${e?.type ?? ''} ${e?.message ?? e}`))
  return queue
}

async function stop() {
  clearInterval(statsTimer)
  statsTimer = null
  pipeline?.destroy()
  call?.close()
  input?.stop()
  pipeline = call = input = null
  lastDecoded = null
  lastRtp = null
  $('flash').hidden = true
  $('local').srcObject = $('remote').srcObject = null
  state.running = false
  $('start').disabled = false
  $('stop').disabled = true
}

let lastDecoded = null
let lastRtp = null
const ms = (v) => (v == null ? '–' : `${v.toFixed(1)} ms`)
async function updateStats(tick = false) {
  if (!call) return
  const p = pipeline
  const [s, rtp] = await Promise.all([p ? p.getStats() : null, callStats(call)])
  if (!call || p !== pipeline) return
  const now = performance.now()
  if (tick && lastDecoded && rtp.video.framesDecoded != null) {
    state.remoteFps = Math.round((rtp.video.framesDecoded - lastDecoded.n) / ((now - lastDecoded.t) / 1000))
  }
  if (tick) lastDecoded = { n: rtp.video.framesDecoded ?? 0, t: now }
  state.stats = s
  state.rtp = rtp
  state.audioStats = p?.getAudioStats() ?? null

  // Latency breakdown over the last interval (the 1 s tick; a manual refresh
  // reuses the last tick's numbers so it doesn't shrink the window).
  if (tick) {
    const rtc = lastRtp ? latencyBreakdown(lastRtp, rtp) : null
    lastRtp = rtp
    if (rtc) {
      const fps = s?.inputFps || input?.stream.getVideoTracks()[0]?.getSettings().frameRate || 30
      const captureMs = 1000 / fps
      const longpipeMs = s && s.latencyMs > 0 ? s.latencyMs : null
      const parts = [captureMs, longpipeMs, rtc.encodeMs, rtc.jitterMs, rtc.decodeMs]
      state.latency = { captureMs, longpipeMs, ...rtc, totalMs: parts.reduce((a, v) => a + (v ?? 0), 0) }
    }
  }
  renderLatency()

  $('r-frames').textContent = `${rtp.video.framesSent ?? 0} / ${rtp.video.framesDecoded ?? 0} (${state.remoteFps} fps, ${rtp.video.frameWidth ?? 0}×${rtp.video.frameHeight ?? 0})`
  $('r-audio').textContent = rtp.audio.packetsReceived != null
    ? `${rtp.audio.packetsReceived} packets, level ${(rtp.audio.audioLevel ?? 0).toFixed(3)}`
    : 'no audio track'

  if (!s) {
    // Bypassed: no pipeline, the raw tracks are on the senders.
    for (const id of ['r-backend', 'r-fps', 'r-denoise', 'r-paths', 'r-restarts']) $(id).textContent = '–'
    $('r-status').textContent = 'bypassed (raw camera + mic → pc1)'
    $('r-model').textContent = 'none (bypass)'
    return
  }
  $('r-status').textContent = s.status
  $('r-backend').textContent = s.backend
  // What's actually running: the worker's model name (changes with setPreset
  // and the adaptive controller), the dtype the backend resolved, the backend.
  // ('none' = the worker didn't answer this time: keep the last name it gave.)
  if (s.preset !== 'none') state.livePreset = s.preset
  const live = state.livePreset === 'boot' ? 'loading…' : state.livePreset ?? state.ready?.preset ?? '…'
  const adaptive = state.model === 'auto' ? ' (auto: adaptive)' : ''
  $('r-model').textContent = s.backend === 'none' ? 'none (passthrough)' : `${live}${adaptive} · ${state.ready?.dtype ?? '…'} · ${s.backend}`
  $('r-fps').textContent = `${Math.round(s.inputFps ?? 0)} / ${s.fps}`
  const a = state.audioStats
  $('r-denoise').textContent = a ? `${a.model}, ${a.active ? 'active' : 'bypassed'}, ${a.sampleRate} Hz` : (state.config?.denoise ? 'starting / unavailable' : 'off')
  $('r-paths').textContent = `${s.inputPath} → ${s.outputPath}`
  $('r-restarts').textContent = String(s.restarts)
}

function renderLatency() {
  const l = state.latency
  $('l-capture').textContent = l ? `≈ ${ms(l.captureMs)} (one frame interval)` : '–'
  $('l-longpipe').textContent = state.bypass ? 'bypassed' : l ? ms(l.longpipeMs) : '–'
  $('l-encode').textContent = l ? ms(l.encodeMs) : '–'
  $('l-jitter').textContent = l ? ms(l.jitterMs) : '–'
  $('l-decode').textContent = l ? ms(l.decodeMs) : '–'
  $('l-total').textContent = l ? `≈ ${ms(l.totalMs)}` : '–'
}

// ── Controls ────────────────────────────────────────────────────────────────

$('start').addEventListener('click', () => { start().catch(() => {}) })
$('stop').addEventListener('click', () => { void stop() })
form.elements.bg.addEventListener('change', async () => {
  if (!pipeline) return
  // Live, no new track. On failure the previous background stays and the call rejects.
  pipeline.setBackground(await backgroundFor(readConfig())).catch((e) => log(`setBackground: ${e.type}`))
})
form.elements.color.addEventListener('input', () => {
  if (pipeline && form.elements.bg.value === 'color') pipeline.setBackground({ color: form.elements.color.value }).catch(() => {})
})
form.elements.denoise.addEventListener('change', () => pipeline?.setDenoise(form.elements.denoise.value === 'on'))
form.elements.backend.addEventListener('change', () => serial('restart', restartPipeline))
form.elements.model.addEventListener('change', () => serial('model switch', switchModel))
form.elements.bypass.addEventListener('change', () => serial('bypass', () => setBypass(form.elements.bypass.checked)))
form.elements.listen.addEventListener('change', () => {
  $('remote').muted = !form.elements.listen.checked   // off by default: loopback + speakers = echo
  pipeline?.resumeAudio()                            // a gesture: unblock a suspended AudioContext
})

// ── Test hooks (used by tests/e2e; harmless otherwise) ──────────────────────

// Average RGB of small patches of the remote (or local) video, at fractional (x, y) spots.
function sampleRemote(points, id = 'remote') {
  const v = $(id)
  if (!v.videoWidth) return null
  const c = document.createElement('canvas')
  c.width = v.videoWidth
  c.height = v.videoHeight
  const g = c.getContext('2d', { willReadFrequently: true })
  g.drawImage(v, 0, 0)
  const out = {}
  for (const [name, [fx, fy]] of Object.entries(points)) {
    const s = Math.max(4, Math.round(c.width * 0.03))
    const x = Math.min(c.width - s, Math.max(0, Math.round(fx * c.width - s / 2)))
    const y = Math.min(c.height - s, Math.max(0, Math.round(fy * c.height - s / 2)))
    const d = g.getImageData(x, y, s, s).data
    let r = 0, gr = 0, b = 0
    for (let i = 0; i < d.length; i += 4) { r += d[i]; gr += d[i + 1]; b += d[i + 2] }
    const n = d.length / 4
    out[name] = [Math.round(r / n), Math.round(gr / n), Math.round(b / n)]
  }
  return { width: c.width, height: c.height, patches: out }
}

// RMS level (0..1) of a stream's audio, measured with WebAudio over ~200 ms.
// RTP audioLevel depends on playout (0 in headless browsers with muted
// output), so the tests measure the decoded signal directly.
let meterCtx = null
async function rmsLevel(stream, ms = 200) {
  const track = stream?.getAudioTracks()[0]
  if (!track) return null
  meterCtx ??= new AudioContext()
  if (meterCtx.state === 'suspended') await meterCtx.resume().catch(() => {})
  const src = meterCtx.createMediaStreamSource(new MediaStream([track]))
  const an = meterCtx.createAnalyser()
  an.fftSize = 2048
  src.connect(an)
  const buf = new Float32Array(an.fftSize)
  let peak = 0
  const end = performance.now() + ms
  while (performance.now() < end) {
    await new Promise((r) => setTimeout(r, 20))
    an.getFloatTimeDomainData(buf)
    let sum = 0
    for (const v of buf) sum += v * v
    peak = Math.max(peak, Math.sqrt(sum / buf.length))
  }
  src.disconnect()
  return peak
}

window.example = {
  state,
  start,
  stop,
  restartPipeline,
  // Same as using the controls (the e2e tests and quick scripts drive these).
  setModel(m) { form.elements.model.value = m; return serial('model switch', switchModel) },
  setBypass(on) { form.elements.bypass.checked = on; return serial('bypass', () => setBypass(on)) },
  sampleRemote,
  // Frames rendered by the local <video> (pipeline.stream) in `ms`: MediaStream out, no WebRTC.
  countLocalFrames(ms) {
    const v = $('local')
    return new Promise((resolve) => {
      let n = 0
      const tick = () => { n++; v.requestVideoFrameCallback(tick) }
      v.requestVideoFrameCallback(tick)
      setTimeout(() => resolve(n), ms)
    })
  },
  get pipeline() { return pipeline },
  async audioLevels() {
    return { local: await rmsLevel(pipeline?.stream), remote: await rmsLevel(call?.remoteStream) }
  },
  get call() { return call },
  async refreshStats() { await updateStats(); return state },
}
