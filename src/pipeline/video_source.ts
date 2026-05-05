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

export async function setupVideoSource(input: VideoSourceInput): Promise<VideoSourceSetup> {
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

  await new Promise<void>((resolve, reject) => {
    video.addEventListener('loadeddata', () => resolve(), { once: true })
    video.addEventListener('error',     () => reject(new Error(`background video: failed to load ${url}`)), { once: true })
  })
  try {
    await video.play()
  } catch (err) {
    throw new Error(`background video: play() rejected — ${(err as Error).message}`)
  }

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
      try {
        const frame = new VideoFrame(video, { timestamp: meta.mediaTime * 1_000_000 })
        mainPort.postMessage({ frame }, [frame])
      } catch {
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
      try {
        const frame = new VideoFrame(video, { timestamp: performance.now() * 1000 })
        mainPort.postMessage({ frame }, [frame])
      } catch {
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
