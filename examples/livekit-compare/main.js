// LiveKit's own background processor vs Longpipe, side by side, on one camera.
//
//   getUserMedia ─┬─ clone → LocalVideoTrack.setProcessor(@livekit/track-processors) → left <video>
//                 └─ clone → LocalVideoTrack.setProcessor(@longpipe/livekit)          → right <video>
//
// No Room or LiveKit server: setProcessor() runs on a LocalVideoTrack on its own,
// and track.attach() shows the processed track. Both processors take the same
// switchTo({ mode }) calls.
//
// URL parameters:
//   ?clean          recording mode: only the two panels (keys 1/2/3/t/r still work)
//   ?preset=xl      Longpipe model tier (xs, small, medium, large, xl)
//   ?mirror=0       don't mirror the video (selfie view is the default)
//   ?input=clip     the e2e test clip instead of the camera (npm run e2e:assets)
//   ?backend=webgl2 force Longpipe's backend (testing; default auto = WebGPU when available)
//   ?bg=blur|background|none, ?touchup=1, ?reframe=1   initial state
import { LocalVideoTrack } from 'livekit-client'
import { BackgroundProcessor as LiveKitProcessor } from '@livekit/track-processors'
import { LongpipeProcessor } from '@longpipe/livekit'
import { makeBackgrounds } from './backgrounds.js'

const $ = (id) => document.getElementById(id)
const params = new URLSearchParams(location.search)
const PRESET = params.get('preset') ?? 'xl'
if (params.has('clean')) document.body.classList.add('clean')
if (params.get('mirror') === '0') document.body.classList.remove('mirror')
$('lk-label').textContent = `(@livekit/track-processors ${__LK_TP_VERSION__})`
$('lp-label').textContent = `(@longpipe/livekit, ${PRESET})`

const state = {
  mode: params.get('bg') ?? 'blur', bgIndex: 0,
  touchup: params.get('touchup') === '1', reframe: params.get('reframe') === '1',
  fps: { livekit: 0, longpipe: 0 }, longpipe: null, livekitFrames: 0, errors: [],
}
let backgrounds = []
let lkTrack, lpTrack, lk, lp

const fail = (msg) => { state.errors.push(msg); $('error').textContent = msg; console.error('[compare]', msg) }

// The same effect for both, in @livekit/track-processors' switchTo() shape.
function modeOptions() {
  if (state.mode === 'blur') return { mode: 'background-blur', blurRadius: 10 }
  if (state.mode === 'background') return { mode: 'virtual-background', imagePath: backgrounds[state.bgIndex].url }
  return { mode: 'disabled' }
}

async function cameraStream() {
  if (params.get('input') === 'clip') {
    const { clipInput } = await import('../vanilla-webrtc/inputs.js')
    return (await clipInput(['/assets/media/woman_on_video_call_360p.mp4', '/assets/media/woman_on_video_call_360p.webm'], { audio: false })).stream
  }
  return navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 30 }, audio: false })
}

async function start() {
  backgrounds = await makeBackgrounds()
  renderThumbs()
  const cam = (await cameraStream()).getVideoTracks()[0]

  // Two independent tracks from one camera, so each processor owns its input.
  lkTrack = new LocalVideoTrack(cam.clone(), undefined, true)
  lpTrack = new LocalVideoTrack(cam.clone(), undefined, true)
  lkTrack.attach($('v-livekit'))
  lpTrack.attach($('v-longpipe'))

  // LiveKit at its best: confidence-mask blending + temporal smoothing (merged
  // upstream in Nov 2026, after 0.8.3; older versions ignore both options).
  lk = LiveKitProcessor({ ...modeOptions(), maskBlending: 'alpha', temporalSmoothing: true, onFrameProcessed: () => state.livekitFrames++ }, 'livekit')
  lp = LongpipeProcessor({
    preset: PRESET,
    ...(params.has('backend') ? { backend: params.get('backend') } : {}),
    background: 'none',
    ...(state.touchup ? { touchup: {} } : {}),
    ...(state.reframe ? { reframe: true } : {}),
    onError: (e) => fail(`Longpipe ${e.type}: ${e.message}`),
  })
  await Promise.all([
    lkTrack.setProcessor(lk).catch((e) => fail(`LiveKit: ${e?.message ?? e}`)),
    lpTrack.setProcessor(lp).then(() => lp.switchTo(modeOptions())),
  ])
  lp.pipeline.ready.then((info) => { state.longpipe = info }, (e) => fail(`Longpipe ready: ${e?.message ?? e}`))
  render()
}

// ── Controls ────────────────────────────────────────────────────────────────

async function setMode(mode) {
  // Pressing Background again cycles the images.
  if (mode === 'background' && state.mode === 'background') state.bgIndex = (state.bgIndex + 1) % backgrounds.length
  state.mode = mode
  render()
  const opts = modeOptions()
  await Promise.all([lk?.switchTo(opts), lp?.switchTo(opts)]).catch((e) => fail(`switch: ${e?.message ?? e}`))
}

async function setBackground(i) {
  state.bgIndex = i
  state.mode = 'blur' // so setMode() below doesn't cycle
  await setMode('background')
}

async function toggle(name) {
  state[name] = !state[name]
  render()
  const p = lp?.pipeline
  if (!p) return
  const call = name === 'touchup' ? p.setTouchup(state.touchup ? {} : null) : p.setReframe(state.reframe ? true : null)
  await call.catch((e) => fail(`${name}: ${e?.message ?? e}`))
}

function renderThumbs() {
  $('thumbs').replaceChildren(...backgrounds.map((b, i) => {
    const el = document.createElement('button')
    el.title = b.name
    el.style.backgroundImage = `url(${b.url})`
    el.addEventListener('click', () => void setBackground(i))
    return el
  }))
}

function render() {
  for (const el of document.querySelectorAll('[data-mode]')) el.setAttribute('aria-pressed', String(el.dataset.mode === state.mode))
  ;[...$('thumbs').children].forEach((el, i) => el.setAttribute('aria-pressed', String(state.mode === 'background' && i === state.bgIndex)))
  $('touchup').setAttribute('aria-pressed', String(state.touchup))
  $('reframe').setAttribute('aria-pressed', String(state.reframe))
  $('chip-touchup').classList.toggle('on', state.touchup)
  $('chip-reframe').classList.toggle('on', state.reframe)
}

for (const el of document.querySelectorAll('[data-mode]')) el.addEventListener('click', () => void setMode(el.dataset.mode))
$('touchup').addEventListener('click', () => void toggle('touchup'))
$('reframe').addEventListener('click', () => void toggle('reframe'))
document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return
  const k = e.key.toLowerCase()
  if (k === '1') void setMode('blur')
  else if (k === '2') void setMode('background')
  else if (k === '3') void setMode('none')
  else if (k === 't') void toggle('touchup')
  else if (k === 'r') void toggle('reframe')
  else if (k === 'c') document.body.classList.toggle('clean')
})

// ── HUD: frames each <video> actually presented per second ──────────────────

function countFrames(video, key) {
  let n = 0
  const tick = () => { n++; video.requestVideoFrameCallback(tick) }
  video.requestVideoFrameCallback(tick)
  let t = performance.now()
  setInterval(() => {
    const now = performance.now()
    state.fps[key] = Math.round((n * 1000) / (now - t))
    n = 0
    t = now
  }, 1000)
}
countFrames($('v-livekit'), 'livekit')
countFrames($('v-longpipe'), 'longpipe')

const BACKEND = { webgpu: 'WebGPU', webgl2: 'WebGL', webgl: 'WebGL', none: 'no GPU' }
setInterval(async () => {
  // MediaPipe's image segmenter with delegate 'GPU' runs on WebGL in the browser.
  $('hud-livekit').textContent = `${state.fps.livekit} fps · WebGL · MediaPipe selfie`
  const s = lp?.pipeline ? await lp.pipeline.getStats().catch(() => null) : null
  state.longpipeStats = s
  $('hud-longpipe').textContent = s
    ? `${state.fps.longpipe} fps · ${BACKEND[s.backend] ?? s.backend} · ${s.preset || PRESET}${s.status === 'running' ? '' : ` · ${s.status}`}`
    : 'starting…'
}, 1000)

// ── Test hooks ──────────────────────────────────────────────────────────────

// Average RGB of small patches of a panel's <video> at fractional (x, y) spots.
function sample(side, points) {
  const v = $(`v-${side}`)
  if (!v.videoWidth) return null
  const c = document.createElement('canvas')
  c.width = v.videoWidth
  c.height = v.videoHeight
  const g = c.getContext('2d', { willReadFrequently: true })
  g.drawImage(v, 0, 0)
  const out = {}
  for (const [name, [fx, fy]] of Object.entries(points)) {
    const s = Math.max(4, Math.round(c.width * 0.04))
    const x = Math.min(c.width - s, Math.max(0, Math.round(fx * c.width - s / 2)))
    const y = Math.min(c.height - s, Math.max(0, Math.round(fy * c.height - s / 2)))
    const d = g.getImageData(x, y, s, s).data
    let r = 0, gr = 0, b = 0
    for (let i = 0; i < d.length; i += 4) { r += d[i]; gr += d[i + 1]; b += d[i + 2] }
    out[name] = [r, gr, b].map((t) => Math.round(t / (d.length / 4)))
  }
  return out
}
window.compare = { state, sample, setMode, toggle, hud: () => [$('hud-livekit').textContent, $('hud-longpipe').textContent] }

render()
start().catch((e) => fail(`start: ${e?.message ?? e}`))
