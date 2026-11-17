// Main-side <video> manager for video background sources.
//
// Accepts a URL, Blob, or existing HTMLVideoElement. Normalizes by always
// constructing an internally-owned hidden <video> (configured: muted,
// looped, playing) — the user's element if they passed one is left
// untouched (we extract its src). Runs an rVFC loop that posts a fresh
// VideoFrame per video-frame to a MessagePort; the worker uploads each
// to the bg tensor on receipt.
//
// The bg pump is fully decoupled from the render pipe: bg updates at the
// source video's framerate regardless of camera fps. No backpressure /
// queue — the worker just overwrites the bg tensor on each arrival, and
// the render loop reads whatever's there at composite time.

export interface VideoSourceSetup {
  port:         MessagePort       // hand to worker via transferList
  transferList: Transferable[]
  cleanup:      () => void
}

export type VideoSourceInput = HTMLVideoElement | string | Blob

type RVFCMeta = { mediaTime: number }
type RVFCCallback = (now: number, meta: RVFCMeta) => void
type RVFCMethod = (cb: RVFCCallback) => number

export interface VideoSourceOptions {
  timeoutMs?: number                     // max wait for the first frame
  onError?:   (message: string) => void  // decode/network error after setup
}

export async function setupVideoSource(input: VideoSourceInput, opts: VideoSourceOptions = {}): Promise<VideoSourceSetup> {
  // Resolve to a URL string. We always create our own <video> rather
  // than mutating the user's element — predictable lifecycle, no
  // surprise side effects on their playback state.
  let url: string
  let ownedObjectURL = false
  if (input instanceof HTMLVideoElement) {
    url = input.currentSrc || input.src
    if (!url) throw new Error('background video: HTMLVideoElement has no src')
  } else if (input instanceof Blob) {
    url = URL.createObjectURL(input)
    ownedObjectURL = true
  } else if (typeof input === 'string') {
    url = input
  } else {
    throw new Error('background video: unsupported input shape')
  }

  // crossOrigin must be set BEFORE src for it to take effect on the
  // network request. 'anonymous' avoids credential leakage and lets us
  // construct VideoFrames from cross-origin video (otherwise the canvas
  // would be tainted).
  const video = document.createElement('video')
  video.crossOrigin = 'anonymous'
  video.muted       = true
  video.loop        = true
  video.playsInline = true
  video.src         = url

  const fail = (msg: string) => {
    video.removeAttribute('src')
    video.load()
    if (ownedObjectURL) URL.revokeObjectURL(url)
    return new Error(msg)
  }
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = opts.timeoutMs
        ? setTimeout(() => reject(new Error(`background video: timed out loading ${url}`)), opts.timeoutMs)
        : null
      video.addEventListener('loadeddata', () => { if (timer) clearTimeout(timer); resolve() }, { once: true })
      video.addEventListener('error',     () => { if (timer) clearTimeout(timer); reject(new Error(`background video: failed to load ${url}`)) }, { once: true })
    })
  } catch (err) {
    throw fail((err as Error).message)
  }
  try {
    await video.play()
  } catch (err) {
    throw fail(`background video: play() rejected — ${(err as Error).message}`)
  }
  // After setup: a decode / network error stops new frames; the worker keeps
  // the last one. Report it (non-fatal) instead of failing silently.
  video.addEventListener('error', () => {
    if (!stopped) opts.onError?.(`background video stopped: ${video.error?.message || 'decode/network error'} (${url})`)
  })

  const channel    = new MessageChannel()
  const mainPort   = channel.port1
  const workerPort = channel.port2

  let stopped = false

  const rvfc = (video as unknown as { requestVideoFrameCallback?: RVFCMethod })
    .requestVideoFrameCallback
    ?.bind(video)

  if (rvfc) {
    const tick: RVFCCallback = (_now, meta) => {
      if (stopped) return
      let frame: VideoFrame | null = null
      try {
        frame = new VideoFrame(video, { timestamp: meta.mediaTime * 1_000_000 })
        mainPort.postMessage({ frame }, [frame])
      } catch {
        frame?.close()
        // VideoFrame() can fail if the underlying media frame isn't
        // ready for this tick (rare but possible). Skip; next rVFC
        // invocation will retry.
      }
      rvfc(tick)
    }
    rvfc(tick)
  } else {
    // rAF fallback for browsers without rVFC on <video>. Less precise
    // (couples to display refresh, not video decode) but functional.
    const tick = () => {
      if (stopped) return
      let frame: VideoFrame | null = null
      try {
        frame = new VideoFrame(video, { timestamp: performance.now() * 1000 })
        mainPort.postMessage({ frame }, [frame])
      } catch {
        frame?.close()
        // skip — see above
      }
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }

  const cleanup = () => {
    stopped = true
    mainPort.close()
    video.pause()
    video.removeAttribute('src')
    video.load()
    if (ownedObjectURL) URL.revokeObjectURL(url)
  }

  return {
    port:         workerPort,
    transferList: [workerPort],
    cleanup,
  }
}
