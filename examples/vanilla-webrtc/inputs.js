// Input streams for the example.
//
//   cameraInput()  — the real thing: getUserMedia video + audio.
//   clipInput(url) — no camera needed: a looping video clip drawn into a
//                    <canvas> and captured, plus a WebAudio tone + noise. Works
//                    the same in every browser (Safari has no
//                    HTMLVideoElement.captureStream), and is what the e2e
//                    tests use. Must be called from a user gesture (the
//                    AudioContext needs one in most browsers).

// facingMode: 'user' (front) / 'environment' (back) on phones; undefined = default camera.
export async function cameraInput({ facingMode } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, frameRate: 30, ...(facingMode ? { facingMode: { ideal: facingMode } } : {}) },
    audio: true,
  })
  return { stream, stop: () => stream.getTracks().forEach((t) => t.stop()) }
}

// Play the first URL this browser can decode (e.g. H.264 MP4, then a VP9 WebM for
// Firefox builds without system H.264 codecs). Throws instead of hanging if none plays.
async function playFirst(urls) {
  for (const url of urls) {
    const video = document.createElement('video')
    video.muted = true
    video.loop = true
    video.playsInline = true
    video.crossOrigin = 'anonymous'
    video.src = url
    const ok = await Promise.race([
      video.play().then(() => true, () => false),
      new Promise((r) => video.addEventListener('error', () => r(false), { once: true })),
      new Promise((r) => setTimeout(() => r(false), 10_000)),
    ])
    if (ok) return video
    video.removeAttribute('src')
  }
  throw new Error(`test clip: none of ${urls.join(', ')} plays in this browser`)
}

export async function clipInput(url, { fps = 30, audio = true } = {}) {
  const video = await playFirst(Array.isArray(url) ? url : [url])
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }))

  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')
  const draw = () => ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  draw()
  // A timer, not requestAnimationFrame: rAF stops in background tabs.
  const timer = setInterval(draw, 1000 / fps)
  const stream = canvas.captureStream(fps)

  let ac = null
  if (audio) {
    // 440 Hz tone + white noise: the denoiser should keep the first and cut
    // the second. Mono, 48 kHz (the denoiser's native rate).
    ac = new AudioContext({ sampleRate: 48000 })
    const dest = ac.createMediaStreamDestination()
    const tone = ac.createOscillator()
    tone.frequency.value = 440
    const toneGain = ac.createGain()
    toneGain.gain.value = 0.2
    tone.connect(toneGain).connect(dest)
    const noiseBuf = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate)
    const ch = noiseBuf.getChannelData(0)
    for (let i = 0; i < ch.length; i++) ch[i] = Math.random() * 2 - 1
    const noise = ac.createBufferSource()
    noise.buffer = noiseBuf
    noise.loop = true
    const noiseGain = ac.createGain()
    noiseGain.gain.value = 0.08
    noise.connect(noiseGain).connect(dest)
    tone.start()
    noise.start()
    if (ac.state === 'suspended') await ac.resume().catch(() => {})
    stream.addTrack(dest.stream.getAudioTracks()[0])
  }

  return {
    stream,
    stop() {
      clearInterval(timer)
      stream.getTracks().forEach((t) => t.stop())
      video.pause()
      video.removeAttribute('src')
      video.load()
      ac?.close()
    },
  }
}

// A generated background picture (no image file to ship or license).
export async function generatedBackgroundImage(w = 1280, h = 720) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')
  const grad = g.createLinearGradient(0, 0, w, h)
  grad.addColorStop(0, '#1d3557')
  grad.addColorStop(1, '#2a9d8f')
  g.fillStyle = grad
  g.fillRect(0, 0, w, h)
  g.fillStyle = 'rgba(255,255,255,0.12)'
  for (let i = 0; i < 12; i++) {
    g.beginPath()
    g.arc(((i * 0.37) % 1) * w, ((i * 0.61) % 1) * h, 40 + ((i * 53) % 120), 0, Math.PI * 2)
    g.fill()
  }
  return createImageBitmap(c)
}

// Glass-to-glass helper (?flash=1): redraws an input's video into a canvas and
// paints a square in its top-left corner, white for `flashMs` once per
// `periodMs`, black otherwise. onFlash(on) fires on the frame where it changes,
// so the page can light a square on screen at the same moment. Film the screen
// in slow motion: the time between the on-screen square and the square in the
// remote video is the latency after capture. The canvas hop costs about a
// frame of its own; the canvas keeps the size it started with (don't rotate
// the phone in this mode).
export async function withFlashMarker(source, { onFlash, periodMs = 1000, flashMs = 100 } = {}) {
  const video = document.createElement('video')
  video.srcObject = new MediaStream(source.stream.getVideoTracks())
  video.muted = true
  video.playsInline = true
  await video.play()
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }))

  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')
  const box = Math.round(Math.min(canvas.width, canvas.height) * 0.18)
  let lit = null
  let stopped = false
  const draw = () => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const on = performance.now() % periodMs < flashMs
    ctx.fillStyle = on ? '#fff' : '#000'
    ctx.fillRect(0, 0, box, box)
    if (on !== lit) { lit = on; onFlash?.(on) }
  }
  // Draw once per camera frame where the browser says when one arrives.
  let timer = null
  if (video.requestVideoFrameCallback) {
    const tick = () => { if (stopped) return; draw(); video.requestVideoFrameCallback(tick) }
    video.requestVideoFrameCallback(tick)
  } else {
    timer = setInterval(draw, 1000 / 30)
  }
  const stream = new MediaStream([canvas.captureStream(30).getVideoTracks()[0], ...source.stream.getAudioTracks()])

  return {
    stream,
    stop() {
      stopped = true
      clearInterval(timer)
      stream.getVideoTracks().forEach((t) => t.stop())
      video.srcObject = null
      onFlash?.(false)
      source.stop()
    },
  }
}
