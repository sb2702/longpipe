// Camera-restart soak (test hook; used by tests/e2e/camera-restart.e2e.mjs and
// tests/e2e/browserstack/ios-camera.mjs). Reproduces the pattern from a user
// report on iOS Safari: camera off (pipeline parked on a placeholder canvas
// stream via setInputStream), camera on (setInputStream with a new camera
// track), and background changes. After each step it measures what the page
// actually SHOWS from pipeline.stream:
//   - delivered fps: frames presented by the local <video> (requestVideoFrameCallback
//     presentedFrames), not getStats().fps (frames handed on);
//   - orientation: the output's aspect vs the input as a <video> displays it
//     (upright), and which rotation of the input the output's content matches.

const GRID = 24   // long side of the comparison grids

// A placeholder like an app shows while the camera is off: a landscape canvas, redrawn on a timer.
export function placeholderStream(w = 640, h = 360, fps = 15) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')
  let n = 0
  const draw = () => { g.fillStyle = '#333'; g.fillRect(0, 0, w, h); g.fillStyle = '#888'; g.fillRect((n++ * 7) % w, h / 3, 40, h / 3) }
  draw()
  const timer = setInterval(draw, 1000 / fps)
  const stream = c.captureStream(fps)
  return { stream, stop() { clearInterval(timer); stream.getTracks().forEach((t) => t.stop()) } }
}

// Luma grid of a <video>'s current frame (as displayed: upright), zero-mean / unit-std.
function lumaGrid(video) {
  const vw = video.videoWidth, vh = video.videoHeight
  if (!vw || !vh) return null
  const s = GRID / Math.max(vw, vh)
  const w = Math.max(2, Math.round(vw * s)), h = Math.max(2, Math.round(vh * s))
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d', { willReadFrequently: true })
  g.drawImage(video, 0, 0, w, h)
  const d = g.getImageData(0, 0, w, h).data
  const data = new Float32Array(w * h)
  let mean = 0
  for (let i = 0; i < w * h; i++) { data[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; mean += data[i] }
  mean /= w * h
  let v = 0
  for (let i = 0; i < w * h; i++) v += (data[i] - mean) ** 2
  const std = Math.sqrt(v / (w * h))
  for (let i = 0; i < w * h; i++) data[i] = std > 2 ? (data[i] - mean) / std : 0
  return { w, h, data, std, mean }
}

// Correlation of `out` with `ref` rotated by `rot` (deg, clockwise) and optionally
// mirrored, over the central 60% of `out` (the subject; backgrounds may be replaced).
function corrAt(ref, out, rot, mirror) {
  const a = [], b = []
  for (let y = Math.floor(out.h * 0.2); y < Math.ceil(out.h * 0.8); y++) {
    for (let x = Math.floor(out.w * 0.2); x < Math.ceil(out.w * 0.8); x++) {
      let u = (x + 0.5) / out.w
      const v = (y + 0.5) / out.h
      if (mirror) u = 1 - u
      // out(u, v) = ref rotated clockwise by `rot`: which ref pixel?
      const [ru, rv] = rot === 0 ? [u, v] : rot === 90 ? [v, 1 - u] : rot === 180 ? [1 - u, 1 - v] : [1 - v, u]
      a.push(out.data[y * out.w + x])
      b.push(ref.data[Math.min(ref.h - 1, Math.floor(rv * ref.h)) * ref.w + Math.min(ref.w - 1, Math.floor(ru * ref.w))])
    }
  }
  // Pearson correlation over the region.
  const n = a.length
  if (!n) return 0
  const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n
  let sab = 0, saa = 0, sbb = 0
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2 }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0
}

// Orientation of the output vs the upright input.
//   'upright' | 'upside-down' | 'sideways' | 'inconclusive' (a flat scene, e.g. a dark camera)
export function compareOrientation(refVideo, outVideo) {
  const ref = lumaGrid(refVideo), out = lumaGrid(outVideo)
  if (!ref || !out) return { verdict: 'no-frame' }
  const refPortrait = ref.h > ref.w, outPortrait = out.h > out.w
  const aspect = { in: `${refVideo.videoWidth}x${refVideo.videoHeight}`, out: `${outVideo.videoWidth}x${outVideo.videoHeight}` }
  const corr = {}
  for (const rot of [0, 90, 180, 270]) corr[rot] = Math.max(corrAt(ref, out, rot, false), corrAt(ref, out, rot, true))
  const best = Object.entries(corr).sort((a, b) => b[1] - a[1])[0]
  const r = (x) => Math.round(x * 100) / 100
  const info = { ...aspect, corr: Object.fromEntries(Object.entries(corr).map(([k, v]) => [k, r(v)])), refStd: r(ref.std), outStd: r(out.std) }
  if (refPortrait !== outPortrait && Math.abs(ref.w - ref.h) > 2) return { verdict: 'sideways', ...info }
  if (ref.std < 4 || out.std < 4 || best[1] < 0.25) return { verdict: 'inconclusive', ...info }
  // Same aspect: decide between upright and upside down. A 90/270 best match with
  // an unchanged aspect counts as sideways only when it's clearly better than
  // upright (rotated content squeezed into the old frame size).
  if (corr[90] > corr[0] + 0.3 || corr[270] > corr[0] + 0.3) return { verdict: corr[0] < 0.25 ? 'sideways' : 'upright', ...info }
  if (corr[180] > corr[0] + 0.15) return { verdict: 'upside-down', ...info }
  return { verdict: corr[0] >= 0.25 ? 'upright' : 'inconclusive', ...info }
}

// Frames presented by a <video> over `ms` (requestVideoFrameCallback presentedFrames).
export function deliveredFps(video, ms = 2000) {
  return new Promise((resolve) => {
    if (!video.requestVideoFrameCallback) return resolve(null)
    let first = null, last = null, cbs = 0, done = false
    const tick = (now, meta) => {
      if (done) return
      cbs++
      if (!first) first = { t: now, n: meta.presentedFrames }
      last = { t: now, n: meta.presentedFrames }
      video.requestVideoFrameCallback(tick)
    }
    video.requestVideoFrameCallback(tick)
    setTimeout(() => {
      done = true
      if (!first || last.t === first.t) return resolve(cbs ? cbs / (ms / 1000) : 0)
      resolve(Math.round(((last.n - first.n) / ((last.t - first.t) / 1000)) * 10) / 10)
    }, ms)
  })
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ctx: {
//   pipeline()            current EffectsPipeline
//   input()               current input { stream, stop }
//   setInput(i)           replace the page's input (so Stop stops the right one)
//   openCamera()          a NEW camera input { stream, stop } (getUserMedia, or a new clip)
//   backgrounds           list of setBackground() values to cycle through
//   outVideo()            the <video> showing pipeline.stream
//   cycles, bgChanges, settleMs, measureMs, offMs
// }
export async function cameraRestartSoak(ctx, status) {
  const { cycles = 10, bgChanges = 5, settleMs = 1500, measureMs = 2000, offMs = 500 } = ctx
  const refVideo = document.createElement('video')
  refVideo.muted = true
  refVideo.playsInline = true
  refVideo.style.cssText = 'position:fixed;right:0;bottom:0;width:64px;opacity:0.01;pointer-events:none'
  document.body.append(refVideo)
  const showRef = async (stream) => {
    refVideo.srcObject = new MediaStream(stream.getVideoTracks())
    await refVideo.play().catch(() => {})
    for (let i = 0; i < 50 && !refVideo.videoWidth; i++) await sleep(100)
  }
  const measure = async (step) => {
    await sleep(settleMs)
    const out = ctx.outVideo()
    const [fps, inFps] = await Promise.all([deliveredFps(out, measureMs), deliveredFps(refVideo, measureMs)])
    const o = compareOrientation(refVideo, out)
    let stats = null
    try { stats = await ctx.pipeline().getStats() } catch {}
    const r = {
      step, t: Math.round(performance.now()), deliveredFps: fps, inputFps: inFps, ...o,
      statsFps: stats?.fps ?? null, statsInputFps: stats?.inputFps != null ? Math.round(stats.inputFps) : null,
      status: stats?.status ?? null, backend: stats?.backend ?? null, inputPath: stats?.inputPath ?? null,
      track: ctx.input()?.stream.getVideoTracks()[0]?.getSettings?.() ?? null,
    }
    status.steps.push(r)
    return r
  }

  await showRef(ctx.input().stream)
  await measure('baseline')
  for (let i = 1; i <= cycles; i++) {
    status.phase = `camera cycle ${i}/${cycles}`
    // Camera off: park the pipeline on a placeholder, then stop the camera.
    const ph = placeholderStream()
    await ctx.pipeline().setInputStream(ph.stream)
    ctx.input().stop()
    await sleep(offMs)
    // Camera on: a new camera track, then drop the placeholder.
    status.phase = `camera cycle ${i}/${cycles}: opening camera`
    const cam = await ctx.openCamera()
    await ctx.pipeline().setInputStream(cam.stream)
    ctx.setInput(cam)
    ph.stop()
    await showRef(cam.stream)
    await measure(`camera cycle ${i}`)
  }
  for (let j = 1; j <= bgChanges; j++) {
    status.phase = `background ${j}/${bgChanges}`
    const bg = ctx.backgrounds[(j - 1) % ctx.backgrounds.length]
    await ctx.pipeline().setBackground(typeof bg === 'function' ? await bg() : bg)
    await measure(`background ${j} (${typeof bg === 'string' ? bg : typeof bg === 'function' ? 'image' : JSON.stringify(bg)})`)
  }
  refVideo.remove()
  status.phase = 'done'
  status.done = true
  return status
}
