// MediaPipe's selfie segmenter vs Longpipe, side by side, on one camera. No WebRTC or SDKs:
// the segmentation model itself, the way apps like Jitsi and LiveKit use it.
//
//   getUserMedia ─┬─ <video> → MediaPipe ImageSegmenter (selfie_segmenter, GPU delegate,
//                 │            confidence mask) → canvas 2D composite        → left <canvas>
//                 └─ clone → new EffectsPipeline(stream, { preset })          → right <video>
//
// The left side uses the raw confidence mask as alpha, with no smoothing over time:
// what the model gives. Both sides draw the same background image or blur.
//
// URL parameters:
//   ?clean          recording mode: only the two panels (keys 1/2/3/t/r still work)
//   ?preset=xl      Longpipe model tier (xs, small, medium, large, xl)
//   ?mirror=0       don't mirror the video (selfie view is the default)
//   ?input=clip     the e2e test clip instead of the camera (npm run e2e:assets)
//   ?backend=webgl2 force Longpipe's backend (testing; default auto = WebGPU when available)
//   ?bg=blur|background|none, ?touchup=1, ?reframe=1   initial state
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'
import { EffectsPipeline } from 'longpipe'
import { makeBackgrounds } from '../livekit-compare/backgrounds.js'

const MP_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'
const BLUR = 0.012   // blur radius as a fraction of frame width, for the MediaPipe side

const $ = (id) => document.getElementById(id)
const params = new URLSearchParams(location.search)
const PRESET = params.get('preset') ?? 'xl'
if (params.has('clean')) document.body.classList.add('clean')
if (params.get('mirror') === '0') document.body.classList.remove('mirror')
$('mp-label').textContent = `(selfie segmenter, tasks-vision ${__MP_VERSION__})`
$('lp-label').textContent = `(${PRESET})`

const state = {
  mode: params.get('bg') ?? 'blur', bgIndex: 0,
  touchup: params.get('touchup') === '1', reframe: params.get('reframe') === '1',
  fps: { mediapipe: 0, longpipe: 0 }, errors: [],
}
let backgrounds = []
let bgImage = null   // the current background as an <img>, for the MediaPipe side
let pipeline

const fail = (msg) => { state.errors.push(msg); $('error').textContent = msg; console.error('[compare]', msg) }

async function cameraStream() {
  if (params.get('input') === 'clip') {
    const { clipInput } = await import('../vanilla-webrtc/inputs.js')
    return (await clipInput(['/assets/media/woman_on_video_call_360p.mp4', '/assets/media/woman_on_video_call_360p.webm'], { audio: false })).stream
  }
  return navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 30 }, audio: false })
}

const longpipeBackground = () =>
  state.mode === 'blur' ? 'blur' : state.mode === 'background' ? backgrounds[state.bgIndex].url : 'none'

function loadBgImage() {
  const img = new Image()
  img.src = backgrounds[state.bgIndex].url
  img.decode().then(() => { bgImage = img }, () => fail('background image failed to load'))
}

// ── MediaPipe side ──────────────────────────────────────────────────────────

async function startMediaPipe(stream) {
  const video = document.createElement('video')
  video.muted = true
  video.playsInline = true
  video.srcObject = stream
  await video.play()

  const fileset = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${__MP_VERSION__}/wasm`)
  const segmenter = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  })

  const out = $('v-mediapipe')
  const ctx = out.getContext('2d')
  const mask = document.createElement('canvas')
  const mctx = mask.getContext('2d')
  const layer = document.createElement('canvas')   // the person, cut out
  const lctx = layer.getContext('2d')
  let frames = 0
  setInterval(() => { state.fps.mediapipe = frames; frames = 0 }, 1000)

  const draw = (now) => {
    const w = video.videoWidth, h = video.videoHeight
    if (w && h) {
      if (out.width !== w || out.height !== h) { out.width = layer.width = w; out.height = layer.height = h }
      segmenter.segmentForVideo(video, now, (result) => {
        const m = result.confidenceMasks?.[0]   // one mask for selfie_segmenter: the person
        if (m) {
          if (mask.width !== m.width || mask.height !== m.height) { mask.width = m.width; mask.height = m.height }
          const conf = m.getAsFloat32Array()
          const img = mctx.createImageData(m.width, m.height)
          for (let i = 0; i < conf.length; i++) img.data[i * 4 + 3] = Math.round(conf[i] * 255)
          mctx.putImageData(img, 0, 0)
        }
        result.close()
      })

      if (state.mode === 'none') {
        ctx.drawImage(video, 0, 0, w, h)
      } else {
        // Background: the blurred frame or the image, cover-fitted.
        if (state.mode === 'blur') {
          ctx.filter = `blur(${Math.round(BLUR * w)}px)`
          ctx.drawImage(video, 0, 0, w, h)
          ctx.filter = 'none'
        } else if (bgImage) {
          const s = Math.max(w / bgImage.width, h / bgImage.height)
          const bw = bgImage.width * s, bh = bgImage.height * s
          ctx.drawImage(bgImage, (w - bw) / 2, (h - bh) / 2, bw, bh)
        }
        // Person: the frame, masked by the confidence mask (bilinear-upscaled by drawImage).
        lctx.globalCompositeOperation = 'copy'
        lctx.drawImage(video, 0, 0, w, h)
        lctx.globalCompositeOperation = 'destination-in'
        lctx.drawImage(mask, 0, 0, w, h)
        ctx.drawImage(layer, 0, 0)
      }
      frames++
    }
    video.requestVideoFrameCallback(draw)
  }
  video.requestVideoFrameCallback(draw)
}

// ── Longpipe side ───────────────────────────────────────────────────────────

async function startLongpipe(stream) {
  pipeline = new EffectsPipeline(stream, {
    preset: PRESET,
    ...(params.has('backend') ? { backend: params.get('backend') } : {}),
    background: longpipeBackground(),
    ...(state.touchup ? { touchup: {} } : {}),
    ...(state.reframe ? { reframe: true } : {}),
    onError: (e) => fail(`Longpipe ${e.type}: ${e.message}`),
    onTrackChanged: () => { $('v-longpipe').srcObject = pipeline.stream },   // stream is updated in place; re-attach
  })
  $('v-longpipe').srcObject = pipeline.stream
  await pipeline.ready.catch((e) => fail(`Longpipe ready: ${e?.message ?? e}`))
}

async function start() {
  backgrounds = await makeBackgrounds()
  loadBgImage()
  renderThumbs()
  const cam = (await cameraStream()).getVideoTracks()[0]
  await Promise.all([
    startMediaPipe(new MediaStream([cam.clone()])).catch((e) => fail(`MediaPipe: ${e?.message ?? e}`)),
    startLongpipe(new MediaStream([cam.clone()])),
  ])
  render()
}

// ── Controls ────────────────────────────────────────────────────────────────

async function setMode(mode) {
  if (mode === 'background' && state.mode === 'background') state.bgIndex = (state.bgIndex + 1) % backgrounds.length
  state.mode = mode
  if (mode === 'background') loadBgImage()
  render()
  await pipeline?.setBackground(longpipeBackground()).catch((e) => fail(`switch: ${e?.message ?? e}`))
}

async function setBackground(i) {
  state.bgIndex = i
  state.mode = 'blur'
  await setMode('background')
}

async function toggle(name) {
  state[name] = !state[name]
  render()
  if (!pipeline) return
  const call = name === 'touchup' ? pipeline.setTouchup(state.touchup ? {} : null) : pipeline.setReframe(state.reframe ? true : null)
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

// ── HUD ─────────────────────────────────────────────────────────────────────

{
  let n = 0, t = performance.now()
  const v = $('v-longpipe')
  const tick = () => { n++; v.requestVideoFrameCallback(tick) }
  v.requestVideoFrameCallback(tick)
  setInterval(() => { const now = performance.now(); state.fps.longpipe = Math.round((n * 1000) / (now - t)); n = 0; t = now }, 1000)
}
const BACKEND = { webgpu: 'WebGPU', webgl2: 'WebGL', webgl: 'WebGL', none: 'no GPU' }
setInterval(async () => {
  $('hud-mediapipe').textContent = `${state.fps.mediapipe} fps · WebGL · 256×256 mask`
  const s = pipeline ? await pipeline.getStats().catch(() => null) : null
  $('hud-longpipe').textContent = s
    ? `${state.fps.longpipe} fps · ${BACKEND[s.backend] ?? s.backend} · ${s.preset || PRESET}${s.status === 'running' ? '' : ` · ${s.status}`}`
    : 'starting…'
}, 1000)

window.compare = { state, setMode, toggle, hud: () => [$('hud-mediapipe').textContent, $('hud-longpipe').textContent] }

render()
start().catch((e) => fail(`start: ${e?.message ?? e}`))
