// Measures the input (camera) frame rate on the main thread, independent of
// how fast the GPU worker keeps up. Used by the adaptive controller (so a
// 15 fps camera isn't mistaken for an overloaded GPU) and by the stall
// watchdog (input advancing while worker output doesn't).
//
// Sources, best first:
//   1. rVFC path: the hidden <video>'s presentedFrames counter.
//   2. MediaStreamTrack.stats.totalFrames (Chromium, camera tracks).
//   3. track.getSettings().frameRate (the negotiated rate, not measured).

export class InputRateMeter {
  private samples: { t: number; frames: number }[] = []

  constructor(
    private getTrack:    () => MediaStreamTrack | undefined,
    private getFramesIn: () => (() => number) | undefined,
  ) {}

  // Total frames delivered by the input so far, or null if unmeasurable.
  totalFrames(): number | null {
    const counter = this.getFramesIn()
    if (counter) return counter()
    const stats = (this.getTrack() as unknown as { stats?: { totalFrames?: number } } | undefined)?.stats
    return typeof stats?.totalFrames === 'number' ? stats.totalFrames : null
  }

  // Call periodically (~1 s). Keeps a short rolling window.
  sample(now = performance.now()): void {
    const frames = this.totalFrames()
    if (frames === null) { this.samples = []; return }
    this.samples.push({ t: now, frames })
    while (this.samples.length > 1 && now - this.samples[0].t > 3_000) this.samples.shift()
  }

  // Measured fps over the window; falls back to the negotiated rate.
  fps(): number | null {
    const a = this.samples[0], b = this.samples[this.samples.length - 1]
    if (a && b && b.t - a.t >= 900 && b.frames >= a.frames) {
      return ((b.frames - a.frames) * 1000) / (b.t - a.t)
    }
    const rate = this.getTrack()?.getSettings?.().frameRate
    return typeof rate === 'number' && rate > 0 ? rate : null
  }

  reset(): void { this.samples = [] }
}
