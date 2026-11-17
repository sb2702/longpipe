// Longpipe in a plain WebRTC call:
//
//   getUserMedia → EffectsPipeline → pipeline.stream → pc1.addTrack → pc2 → <video>
//
// Everything Longpipe-specific is in startPipeline() and the three callbacks
// (onError, onRecovered, onTrackChanged). The rest is ordinary WebRTC + UI.
//
// URL parameters preset the controls (handy on a phone, and used by the e2e
// tests): ?input=clip&bg=color&color=%2300ff00&denoise=on&backend=webgl2
// plus three that have no control:
//   preset=fast|balanced|quality  model preset (default 'auto' = benchmark the device)
//   weights=local|<url> where weights come from ('local' = ./assets/models/,
//                       see `npm run e2e:assets`; default: Longpipe's CDN)
//   denoiseModel=rnnoise|dfn|…  (default 'auto')
//   debug=1             verbose [longpipe/…] console logging
import { EffectsPipeline } from 'longpipe'
import { startLoopbackCall, callStats } from './call.js'
import { cameraInput, clipInput, generatedBackgroundImage } from './inputs.js'

const $ = (id) => document.getElementById(id)
const form = $('controls')
const params = new URLSearchParams(location.search)
for (const name of ['input', 'bg', 'color', 'denoise', 'backend']) {
  if (params.has(name)) form.elements[name].value = params.get(name)
}
const CLIP_URL = './assets/media/woman_on_video_call_360p.mp4'
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
  stats: null,          // last pipeline.getStats()
  audioStats: null,     // last pipeline.getAudioStats()
  rtp: null,            // last callStats()
  remoteFps: 0,
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
    preset: params.get('preset') ?? 'auto',
    denoiseModel: params.get('denoiseModel') ?? 'auto',
  }
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
    preset: cfg.preset,
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
  const cfg = readConfig()
  Object.assign(state, { running: true, config: cfg, ready: null, readyError: null, errors: [], recovered: [], trackChanges: 0 })
  $('start').disabled = true
  try {
    input = cfg.input === 'clip' ? await clipInput(CLIP_URL) : await cameraInput()
    pipeline = startPipeline(cfg, input.stream, await backgroundFor(cfg))
    $('local').srcObject = pipeline.stream
    call = await startLoopbackCall(pipeline.stream, {
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

// Backend is a construction option, so changing it means a new pipeline. The
// call stays up: swap the tracks on the existing senders.
async function restartPipeline() {
  if (!state.running) return
  const cfg = readConfig()
  const old = pipeline
  Object.assign(state, { config: cfg, ready: null, readyError: null })
  pipeline = startPipeline(cfg, input.stream, await backgroundFor(cfg))
  await call.senders.video?.replaceTrack(pipeline.stream.getVideoTracks()[0] ?? null)
  await call.senders.audio?.replaceTrack(pipeline.stream.getAudioTracks()[0] ?? null)
  $('local').srcObject = pipeline.stream
  old.destroy()
  log(`pipeline restarted with backend=${cfg.backend}`)
}

async function stop() {
  clearInterval(statsTimer)
  statsTimer = null
  pipeline?.destroy()
  call?.close()
  input?.stop()
  pipeline = call = input = null
  lastDecoded = null
  $('local').srcObject = $('remote').srcObject = null
  state.running = false
  $('start').disabled = false
  $('stop').disabled = true
}

let lastDecoded = null
async function updateStats(tick = false) {
  if (!pipeline || !call) return
  const [s, rtp] = await Promise.all([pipeline.getStats(), callStats(call)])
  if (!pipeline) return
  const now = performance.now()
  if (tick && lastDecoded && rtp.video.framesDecoded != null) {
    state.remoteFps = Math.round((rtp.video.framesDecoded - lastDecoded.n) / ((now - lastDecoded.t) / 1000))
  }
  if (tick) lastDecoded = { n: rtp.video.framesDecoded ?? 0, t: now }
  state.stats = s
  state.rtp = rtp
  state.audioStats = pipeline.getAudioStats()

  $('r-status').textContent = s.status
  $('r-backend').textContent = `${s.backend}${state.ready?.preset ? ` (${state.ready.preset}, ${state.ready.dtype})` : ''}`
  $('r-fps').textContent = `${Math.round(s.inputFps ?? 0)} / ${s.fps}`
  $('r-frames').textContent = `${rtp.video.framesSent ?? 0} / ${rtp.video.framesDecoded ?? 0} (${state.remoteFps} fps, ${rtp.video.frameWidth ?? 0}×${rtp.video.frameHeight ?? 0})`
  $('r-audio').textContent = rtp.audio.packetsReceived != null
    ? `${rtp.audio.packetsReceived} packets, level ${(rtp.audio.audioLevel ?? 0).toFixed(3)}`
    : 'no audio track'
  const a = state.audioStats
  $('r-denoise').textContent = a ? `${a.model}, ${a.active ? 'active' : 'bypassed'}, ${a.sampleRate} Hz` : (state.config?.denoise ? 'starting / unavailable' : 'off')
  $('r-paths').textContent = `${s.inputPath} → ${s.outputPath}`
  $('r-restarts').textContent = String(s.restarts)
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
form.elements.backend.addEventListener('change', () => { restartPipeline().catch((e) => log(`restart failed: ${e?.message ?? e}`)) })
form.elements.listen.addEventListener('change', () => {
  $('remote').muted = !form.elements.listen.checked   // off by default: loopback + speakers = echo
  pipeline?.resumeAudio()                            // a gesture: unblock a suspended AudioContext
})

// ── Test hooks (used by tests/e2e; harmless otherwise) ──────────────────────

// Average RGB of small patches of the REMOTE video, at fractional (x, y) spots.
function sampleRemote(points) {
  const v = $('remote')
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
  sampleRemote,
  get pipeline() { return pipeline },
  async audioLevels() {
    return { local: await rmsLevel(pipeline?.stream), remote: await rmsLevel(call?.remoteStream) }
  },
  get call() { return call },
  async refreshStats() { await updateStats(); return state },
}
