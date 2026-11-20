// Longpipe in a LiveKit call. Two participants join the same room from this page:
//
//   camera → LocalVideoTrack.setProcessor(Longpipe) → publish (alice)
//          → LiveKit server → subscribe (bob) → remote <video>
//
// The Longpipe-specific code is applyVideoProcessor() and applyDenoise(). Everything else is plain LiveKit.
// URL parameters preset the controls (the e2e test uses them):
//   ?input=clip&processor=longpipe&bg=color&denoise=off&backend=none&weights=local&preset=fast&res=360
import { Room, RoomEvent, Track, LocalVideoTrack, LocalAudioTrack, createLocalVideoTrack, createLocalAudioTrack } from 'livekit-client'
import { LongpipeProcessor, DenoiseProcessor, EffectsPipeline } from '@longpipe/livekit'
import { BackgroundProcessor as LiveKitBackgroundProcessor } from '@livekit/track-processors'
import { clipInput, generatedBackgroundImage } from '../vanilla-webrtc/inputs.js'

const $ = (id) => document.getElementById(id)
const form = $('controls')
const params = new URLSearchParams(location.search)
for (const name of ['input', 'processor', 'bg', 'denoise']) if (params.has(name)) form.elements[name].value = params.get(name)
const CLIP_URL = ['/assets/media/woman_on_video_call_360p.mp4', '/assets/media/woman_on_video_call_360p.webm']
const ROOM = params.get('room') ?? `longpipe-${Math.random().toString(36).slice(2, 8)}`
// Test-only knobs: local weights (npm run e2e:assets), backend, model tier.
const LONGPIPE_OPTS = {
  ...(params.get('weights') === 'local' ? { weightsBaseUrl: new URL('/assets/models/', location.href).href } : {}),
  ...(params.has('backend') ? { backend: params.get('backend') } : {}),
  ...(params.get('preset') ? { preset: params.get('preset') } : {}),
}

const state = { running: false, errors: [], ready: null, stats: null, sentFps: 0, remote: null, processor: null }
window.example = { state, sampleRemote: (pts) => sample($('remote'), pts) }
let input, pub, sub, videoTrack, audioTrack, statsTimer
let bgImageUrl

const log = (msg) => { $('log').textContent = `${new Date().toISOString().slice(11, 19)} ${msg}\n` + $('log').textContent; console.log('[example]', msg) }
const cfg = () => Object.fromEntries(['input', 'processor', 'bg', 'denoise'].map((n) => [n, form.elements[n].value]))

async function backgroundImageUrl() {
  if (bgImageUrl) return bgImageUrl
  const bmp = await generatedBackgroundImage()
  const c = new OffscreenCanvas(bmp.width, bmp.height)
  c.getContext('2d').drawImage(bmp, 0, 0)
  return (bgImageUrl = URL.createObjectURL(await c.convertToBlob({ type: 'image/jpeg' })))
}

async function token(identity) {
  const r = await fetch(`/token?identity=${identity}&room=${ROOM}`)
  return r.json()
}

// ── Longpipe ────────────────────────────────────────────────────────────────

async function applyVideoProcessor() {
  const c = cfg()
  const current = videoTrack.getProcessor()
  if (c.processor === 'longpipe') {
    const background = c.bg === 'image' ? await backgroundImageUrl() : c.bg === 'color' ? { color: '#00ff00' } : c.bg
    if (current?.name === 'longpipe') {
      await current.setBackground(background)   // live, same track
    } else {
      const p = LongpipeProcessor({
        background, ...LONGPIPE_OPTS,
        onError: (e) => { state.errors.push({ type: e.type, fatal: e.fatal }); $('error').textContent = `${e.type}: ${e.message}`; log(`onError ${e.type}`) },
      })
      await videoTrack.setProcessor(p)
      p.pipeline.ready.then((info) => { state.ready = info; log(`ready ${JSON.stringify(info)}`) }, (e) => log(`ready rejected: ${e.type}`))
    }
  } else {
    // LiveKit's own MediaPipe processor, for comparison. It has no colour mode.
    const mode = c.bg === 'blur' ? { mode: 'background-blur', blurRadius: 10 }
      : c.bg === 'none' ? { mode: 'disabled' } : { mode: 'virtual-background', imagePath: await backgroundImageUrl() }
    if (current?.name === 'livekit') await current.switchTo(mode)
    else await videoTrack.setProcessor(LiveKitBackgroundProcessor(mode, 'livekit'))
  }
  state.processor = c.processor
}

// Audio processors need the room's AudioContext, which LiveKit sets on publish:
// call this after publishTrack(audioTrack).
async function applyDenoise() {
  const on = cfg().denoise === 'on'
  const current = audioTrack.getProcessor()
  if (on && !current) {
    await audioTrack.setProcessor(DenoiseProcessor({
      weightsBaseUrl: LONGPIPE_OPTS.weightsBaseUrl,
      model: params.get('denoiseModel') ?? 'auto',
      onError: (msg) => log(`denoise: ${msg}`),
    }))
  }
  if (!on && current) await audioTrack.stopProcessor()
}

// ── Call ────────────────────────────────────────────────────────────────────

async function join() {
  EffectsPipeline.unlockAudio()   // in the click, before any await: lets denoise start
  const c = cfg()
  $('join').disabled = true
  try {
    if (c.input === 'clip') {
      input = await clipInput(CLIP_URL)
      videoTrack = new LocalVideoTrack(input.stream.getVideoTracks()[0], undefined, true)
      audioTrack = new LocalAudioTrack(input.stream.getAudioTracks()[0], undefined, true)
    } else {
      const [width, height] = params.get('res') === '360' ? [640, 360] : [1280, 720]
      videoTrack = await createLocalVideoTrack({ resolution: { width, height, frameRate: 30 } })
      // The browser's own noise suppression off when Longpipe denoises.
      audioTrack = await createLocalAudioTrack({ noiseSuppression: c.denoise !== 'on' })
    }
    videoTrack.attach($('local'))

    sub = new Room()
    sub.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === Track.Kind.Video) { track.attach($('remote')); state.remote = track; log('subscriber: video track') }
    })
    const s = await token('bob')
    await sub.connect(s.url, s.token)

    pub = new Room()
    const p = await token('alice')
    await pub.connect(p.url, p.token)
    await applyVideoProcessor()   // before publishing: the raw camera is never sent
    await pub.localParticipant.publishTrack(videoTrack)
    await pub.localParticipant.publishTrack(audioTrack)
    await applyDenoise()
    state.running = true
    $('leave').disabled = false
    statsTimer = setInterval(updateStats, 1000)
    log(`joined room ${ROOM}: ${JSON.stringify(c)}`)
  } catch (e) {
    log(`join failed: ${e?.message ?? e}`)
    $('error').textContent = String(e?.message ?? e)
    await leave()
  }
}

async function leave() {
  clearInterval(statsTimer)
  await pub?.disconnect()   // stops the tracks, which destroys their processors
  await sub?.disconnect()
  videoTrack?.stop()
  audioTrack?.stop()
  input?.stop()
  pub = sub = videoTrack = audioTrack = input = null
  state.running = false
  $('join').disabled = false
  $('leave').disabled = true
}

let lastSent = null
async function updateStats() {
  if (!videoTrack) return
  const proc = videoTrack.getProcessor()
  const pipeline = proc?.name === 'longpipe' ? proc.pipeline : null
  const s = pipeline ? await pipeline.getStats() : null
  state.stats = s
  // Frames the publisher actually sent, whatever the processor.
  let sent = 0
  for (const r of (await videoTrack.sender?.getStats())?.values() ?? []) if (r.type === 'outbound-rtp') sent += r.framesSent ?? 0
  const now = performance.now()
  if (lastSent) state.sentFps = Math.round((sent - lastSent.n) / ((now - lastSent.t) / 1000))
  lastSent = { n: sent, t: now }
  const rv = $('remote')
  $('r-processor').textContent = proc?.name ?? 'none'
  $('r-backend').textContent = s ? `${s.backend} · ${s.preset}` : proc ? 'MediaPipe (LiveKit)' : '–'
  $('r-fps').textContent = s ? `${Math.round(s.inputFps ?? 0)} / ${s.fps} (sent ${state.sentFps})` : `sent ${state.sentFps}`
  $('r-remote').textContent = `${rv.videoWidth}×${rv.videoHeight}, ${await window.example.remoteFrames()} frames decoded`
  const d = (state.denoise = audioTrack?.getProcessor()?.denoiser?.getStats() ?? null)
  $('r-denoise').textContent = d ? `${d.model}, ${d.active ? 'active' : 'bypassed'}` : 'off'
}

// Average RGB of small patches of a <video> at fractional (x, y) spots (used by the e2e test).
function sample(v, points) {
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
    out[name] = [r, gr, b].map((t) => Math.round(t / (d.length / 4)))
  }
  return { width: c.width, height: c.height, patches: out }
}

$('join').addEventListener('click', () => void join())
$('leave').addEventListener('click', () => void leave())
const onChange = (fn) => () => { if (state.running) fn().catch((e) => log(`switch failed: ${e?.message ?? e}`)) }
form.elements.processor.addEventListener('change', onChange(applyVideoProcessor))
form.elements.bg.addEventListener('change', onChange(applyVideoProcessor))
form.elements.denoise.addEventListener('change', onChange(applyDenoise))
// Frames the subscriber decoded (RTP stats: works the same in every browser).
window.example.remoteFrames = async () => {
  let n = 0
  for (const r of (await state.remote?.receiver?.getStats())?.values() ?? []) if (r.type === 'inbound-rtp') n += r.framesDecoded ?? 0
  return n
}
// Test hook: mute + unmute the camera. LiveKit stops a camera track on mute and
// re-acquires it on unmute, which ends the pipeline's input and calls restart().
window.example.muteCycle = async () => {
  await videoTrack.mute()
  await new Promise((r) => setTimeout(r, 1000))
  await videoTrack.unmute()
}
window.example.processorStats = () => videoTrack?.getProcessor()?.pipeline?.getStats() ?? null
window.example.processedTrackId = () => videoTrack?.getProcessor()?.processedTrack?.id ?? null
// Test hook: swap the source track (what a camera switch does), to exercise processor.restart().
window.example.replaceSource = async () => {
  const next = await clipInput(CLIP_URL, { audio: false })
  const old = input
  input = { stream: next.stream, stop: () => { next.stop(); old?.stop() } }
  await videoTrack.replaceTrack(next.stream.getVideoTracks()[0], true)
}
