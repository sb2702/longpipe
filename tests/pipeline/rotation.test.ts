// Camera rotation end to end: a synthetic camera whose pixels are in "sensor"
// orientation plus a display transform, through the real Pipeline (worker,
// GPU backend, every output transport), checking the OUTPUT track's size and
// orientation with a quadrant test pattern.
//
// Two sources:
//   - native: VideoFrames constructed with { rotation, flip } (Chromium 138+
//     supports the init fields and applies them in its upload paths; the SDK
//     must not rotate again);
//   - hidden: plain frames + TUNING.simulateHiddenOrientation — what shipping
//     Safari does (rotation tagged but not exposed, and ignored by every upload
//     path), measured on main by the orientation probe. The probe itself is
//     unit-tested in orientation_probe.test.ts.
import { describe, it, expect, afterEach, beforeEach } from 'vitest'
import { Pipeline } from '~/pipeline/index'
import { TUNING } from '~/pipeline/tuning'
import type { Orientation } from '~/model/orientation'
import { stubWeights, restoreFetch, errorLog, SMALL, sleep, waitUntil } from '../helpers/pipeline'

let cleanup: (() => void)[] = []
beforeEach(async () => { await stubWeights() })
afterEach(() => { for (const f of cleanup.reverse()) { try { f() } catch {} } cleanup = []; restoreFetch() })

// Upright pattern: top-left quadrant red, top-right green, bottom half blue.
const RED = [230, 30, 30], GREEN = [30, 200, 30], BLUE = [30, 30, 230]
function uprightPattern(w: number, h: number): OffscreenCanvas {
  const c = new OffscreenCanvas(w, h), x = c.getContext('2d')!
  x.fillStyle = 'rgb(30,30,230)'; x.fillRect(0, 0, w, h)
  x.fillStyle = 'rgb(230,30,30)'; x.fillRect(0, 0, w / 2, h / 2)
  x.fillStyle = 'rgb(30,200,30)'; x.fillRect(w / 2, 0, w / 2, h / 2)
  return c
}
// The sensor-orientation image whose display (rotate CW, then flip) is the
// upright pattern: source = rotCW(-r)(flip(upright)).
function sensorImage(sw: number, sh: number, o: Orientation): OffscreenCanvas {
  const sideways = o.rotation === 90 || o.rotation === 270
  const uw = sideways ? sh : sw, uh = sideways ? sw : sh
  const up = uprightPattern(uw, uh)
  const c = new OffscreenCanvas(sw, sh), x = c.getContext('2d')!
  x.translate(sw / 2, sh / 2)
  x.rotate(-o.rotation * Math.PI / 180)
  if (o.flip) x.scale(-1, 1)
  x.drawImage(up, -uw / 2, -uh / 2)
  return c
}

type MstgCtor = new (init: { kind: 'video' }) => MediaStreamTrack & { writable: WritableStream<VideoFrame> }
type MstpCtor = new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

// Camera = MediaStreamTrackGenerator fed at 30 fps. `state` is mutable so a
// test can turn the "phone" mid-stream.
function rotatedCamera(state: { o: Orientation; native: boolean }, sw = 640, sh = 360) {
  const Gen = (self as unknown as { MediaStreamTrackGenerator: MstgCtor }).MediaStreamTrackGenerator
  const track = new Gen({ kind: 'video' })
  const writer = track.writable.getWriter()
  let t = 0, stopped = false
  const cache = new Map<string, OffscreenCanvas>()
  const timer = setInterval(() => {
    if (stopped) return
    const key = `${state.o.rotation}${state.o.flip}`
    let img = cache.get(key)
    if (!img) { img = sensorImage(sw, sh, state.o); cache.set(key, img) }
    const init: VideoFrameInit & { rotation?: number; flip?: boolean } = { timestamp: (t++) * 33_333 }
    if (state.native) { init.rotation = state.o.rotation; init.flip = state.o.flip }
    const f = new VideoFrame(img, init)
    writer.write(f).catch(() => f.close())
  }, 33)
  const stream = new MediaStream([track])
  return { stream, stop: () => { stopped = true; clearInterval(timer); track.stop() } }
}

// Latest output frame → size + mean colour of three probe regions.
async function sampleOutput(track: MediaStreamTrack): Promise<{ w: number; h: number; tl: number[]; tr: number[]; bottom: number[] }> {
  const Proc = (self as unknown as { MediaStreamTrackProcessor: MstpCtor }).MediaStreamTrackProcessor
  const reader = new Proc({ track }).readable.getReader()
  let frame: VideoFrame | undefined
  for (let i = 0; i < 4; i++) {           // skip a few: take a current frame
    const r = await reader.read()
    frame?.close()
    frame = r.value
  }
  void reader.cancel().catch(() => {})
  if (!frame) throw new Error('no output frame')
  const w = frame.displayWidth, h = frame.displayHeight
  const c = new OffscreenCanvas(w, h).getContext('2d')!
  c.drawImage(frame, 0, 0)
  frame.close()
  const mean = (x: number, y: number) => {
    const d = c.getImageData(Math.round(x * w) - 4, Math.round(y * h) - 4, 8, 8).data
    const m = [0, 0, 0]
    for (let i = 0; i < d.length; i += 4) { m[0] += d[i]; m[1] += d[i + 1]; m[2] += d[i + 2] }
    return m.map(v => Math.round(v / (d.length / 4)))
  }
  return { w, h, tl: mean(0.2, 0.2), tr: mean(0.8, 0.2), bottom: mean(0.5, 0.85) }
}

const near = (got: number[], want: number[], tol = 60) => got.every((v, i) => Math.abs(v - want[i]) <= tol)

function expectUpright(s: Awaited<ReturnType<typeof sampleOutput>>, portrait: boolean, what: string) {
  expect(s.h > s.w, `${what}: ${s.w}x${s.h} portrait=${portrait}`).toBe(portrait)
  expect(near(s.tl, RED), `${what}: top-left ${s.tl} should be red`).toBe(true)
  expect(near(s.tr, GREEN), `${what}: top-right ${s.tr} should be green`).toBe(true)
  expect(near(s.bottom, BLUE), `${what}: bottom ${s.bottom} should be blue`).toBe(true)
}

const TOPOLOGIES = [
  { name: 'mstg',             topology: { input: 'mstp', output: 'mstg' } },
  { name: 'bitmap-shuttle',   topology: { input: 'rvfc-postmessage', output: 'bitmap-shuttle' } },
  { name: 'transfer-capture', topology: { input: 'rvfc-postmessage', output: 'transfer-capture' } },
] as const
const BACKENDS = ['webgpu', 'webgl'] as const

function nativeRotationSupported(): boolean {
  try {
    const f = new VideoFrame(new Uint8Array(16), { format: 'RGBA', codedWidth: 2, codedHeight: 2, timestamp: 0, rotation: 90 } as VideoFrameBufferInit)
    const ok = (f as unknown as { rotation?: number }).rotation === 90
    f.close()
    return ok
  } catch { return false }
}

async function startPipeline(stream: MediaStream, backend: 'webgpu' | 'webgl', background: 'blur' | 'none') {
  TUNING.forceBackend = backend
  const log = errorLog()
  // Model on ('blur'): the tier runs (transposed for portrait). Blur keeps
  // the big flat quadrants' colours, so the pattern check still holds.
  const pipeline = new Pipeline(stream, { preset: SMALL, background, onError: log.onError })
  cleanup.push(() => pipeline.destroy())
  const info = await pipeline.ready
  if (info.backend !== (backend === 'webgl' ? 'webgl2' : 'webgpu')) return { pipeline, log, skipped: `backend ${backend} unavailable (got ${info.backend})` }
  return { pipeline, log, skipped: null as string | null }
}

describe.each(BACKENDS)('camera rotation (%s)', (backend) => {
  describe.each(TOPOLOGIES)('$name', ({ name, topology }) => {
    it('native VideoFrame rotation: upright portrait output, follows a mid-stream turn', async (ctx) => {
      if (!nativeRotationSupported()) ctx.skip()   // reason: VideoFrameInit.rotation unsupported in this browser
      TUNING.forceTopology = { ...topology }
      const state = { o: { rotation: 90, flip: false } as Orientation, native: true }
      const cam = rotatedCamera(state)
      cleanup.push(() => cam.stop())
      const { pipeline, log, skipped } = await startPipeline(cam.stream, backend, 'blur')
      if (skipped) { console.warn(`skip (${name}): ${skipped}`); ctx.skip() }
      const out = () => pipeline.stream.getVideoTracks()[0]
      await sleep(800)
      expectUpright(await sampleOutput(out()), true, `${name} rot90`)
      // The phone turns to landscape (other way round = 180 on the sensor).
      state.o = { rotation: 180, flip: false }
      await sleep(1200)
      expectUpright(await sampleOutput(out()), false, `${name} rot180`)
      state.o = { rotation: 270, flip: true }
      await sleep(1200)
      expectUpright(await sampleOutput(out()), true, `${name} rot270+flip`)
      expect(log.errors).toEqual([])
    })

    if (name !== 'mstg') {
      it('hidden (Safari-style) rotation: upright portrait output, follows a mid-stream turn', async (ctx) => {
        TUNING.forceTopology = { ...topology }
        const state = { o: { rotation: 90, flip: false } as Orientation, native: false }
        TUNING.simulateHiddenOrientation = state.o
        const cam = rotatedCamera(state)
        cleanup.push(() => cam.stop())
        const { pipeline, log, skipped } = await startPipeline(cam.stream, backend, 'blur')
        if (skipped) { console.warn(`skip (${name}): ${skipped}`); ctx.skip() }
        const out = () => pipeline.stream.getVideoTracks()[0]
        await sleep(800)
        expectUpright(await sampleOutput(out()), true, `${name} hidden rot90`)
        for (const o of [{ rotation: 0, flip: false }, { rotation: 270, flip: true }, { rotation: 180, flip: true }] as Orientation[]) {
          state.o = o
          TUNING.simulateHiddenOrientation = o
          await sleep(1200)
          expectUpright(await sampleOutput(out()), o.rotation % 180 !== 0, `${name} hidden ${o.rotation}${o.flip ? '+flip' : ''}`)
        }
        // No frame gaps / black frames worth an error, and the pipeline is healthy.
        expect(log.errors).toEqual([])
        expect((await pipeline.getStats()).status).toBe('running')
      })
    }
  })
})

describe('camera rotation with face touch-up geometry inputs', () => {
  it('portrait output with background none (passthrough path) is upright too', async (ctx) => {
    TUNING.forceTopology = { input: 'rvfc-postmessage', output: 'bitmap-shuttle' }
    const state = { o: { rotation: 270, flip: false } as Orientation, native: false }
    TUNING.simulateHiddenOrientation = state.o
    const cam = rotatedCamera(state)
    cleanup.push(() => cam.stop())
    const { pipeline, skipped } = await startPipeline(cam.stream, 'webgpu', 'none')
    if (skipped) ctx.skip()
    await waitUntil(() => pipeline.stream.getVideoTracks()[0].readyState === 'live', 2000, 'live')
    await sleep(800)
    expectUpright(await sampleOutput(pipeline.stream.getVideoTracks()[0]), true, 'passthrough rot270')
  })
})
