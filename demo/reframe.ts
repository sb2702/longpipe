import type { Backend, Dtype, Op } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// ─────────────────────────────────────────────────────────────────────────────
// SCRATCH PAD — auto-reframe v0 logic, deliberately on the CPU.
//
// This is NOT the shape it ships in. Production will keep the state in a carrier
// tensor and apply the crop as a sampling transform in the compositor (no
// readback, both backends). This page exists to find logic that FEELS right
// first; porting it is mechanical once the numbers are settled.
//
// The model (per Sam's read of Meet's behavior):
//   • the frame center is the BASE — reframe is a nudge from it, not a re-centre
//   • the subject applies "gravity" pulling the crop centre toward itself; at
//     gravity < 1 the subject never actually gets centred, it just gets closer
//   • the crop keeps the frame's aspect and shrinks by `zoom`
//   • two hard constraints: the crop must stay INSIDE the frame, and must
//     CONTAIN the subject with margin
//
// Those constraints are why a head in the extreme corner does nothing at all,
// with no special case: containing it would need a crop ≈ the whole frame, i.e.
// zoom ≈ 1, i.e. no visible change. The "does nothing at the corner" behaviour
// and the "moves toward, never centres" behaviour are the same rule.
//
// Modes:
//   auto   — recompute every frame, deadband + ease (hold still, then move
//            deliberately, then hold — a camera operator, not a tracker)
//   manual — solve once on demand and FREEZE, for apps copying Meet's button
// ─────────────────────────────────────────────────────────────────────────────

const MASK_W = 64, MASK_H = 40   // alpha downsample for the body-centroid probe

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s }
const numv = (id: string) => parseFloat($<HTMLInputElement>(id).value)
const val = (id: string) => $<HTMLSelectElement>(id).value
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const lerp = (a: number, b: number, t: number) => a + (b - a) * t

interface Rect { cx: number; cy: number; w: number; h: number }
interface Box { cx: number; cy: number; hs: number; score: number }

// Solve the crop for a subject at `target` (frame fractions), whose face spans
// ±halfX/±halfY. Returns null when no zoom ≥ 1 can both contain the subject and
// stay inside the frame — the corner no-op.
//
// `zoomReq` is relaxed toward 1 in steps rather than solved closed-form: the
// clamp makes the centre depend on the crop size, so it's implicit. Steps are
// cheap here and keep the rule legible while we're still deciding what it is.
function solveCrop(
  target: { x: number; y: number }, halfX: number, halfY: number,
  zoomReq: number, gravity: number, margin: number,
): Rect | null {
  for (let t = 0; t <= 1.0001; t += 0.04) {
    const z = lerp(zoomReq, 1, t)
    const w = 1 / z, h = 1 / z            // equal scale in both dims → aspect preserved
    const cx = clamp(0.5 + gravity * (target.x - 0.5), w / 2, 1 - w / 2)
    const cy = clamp(0.5 + gravity * (target.y - 0.5), h / 2, 1 - h / 2)
    const fitsX = Math.abs(target.x - cx) + halfX + margin <= w / 2
    const fitsY = Math.abs(target.y - cy) + halfY + margin <= h / 2
    if (fitsX && fitsY) return { cx, cy, w, h }
  }
  return null
}

// Deadband + ease. Holds until the target escapes the deadband, then eases until
// it's back well inside — the hold/move/hold cadence, rather than continuous
// tracking (which reads as swimming as the subject breathes and shifts).
class Smoother {
  cur: Rect | null = null
  private moving = false

  step(want: Rect, dead: number, ease: number): Rect {
    if (!this.cur) { this.cur = { ...want }; return this.cur }
    const d = Math.hypot(want.cx - this.cur.cx, want.cy - this.cur.cy)
      + Math.abs(want.w - this.cur.w)
    if (d > dead) this.moving = true
    else if (d < dead * 0.3) this.moving = false
    if (this.moving) {
      this.cur.cx = lerp(this.cur.cx, want.cx, ease)
      this.cur.cy = lerp(this.cur.cy, want.cy, ease)
      this.cur.w  = lerp(this.cur.w,  want.w,  ease)
      this.cur.h  = lerp(this.cur.h,  want.h,  ease)
    }
    return this.cur
  }
  reset() { this.cur = null; this.moving = false }
  get isMoving() { return this.moving }
}

async function createBackend(name: string, dtype: Dtype, canvas: OffscreenCanvas): Promise<Backend> {
  if (name === 'webgpu') {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    if (dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) throw new Error('adapter lacks shader-f16')
    return WebGPUBackend.create({ canvas, dtype })
  }
  return WebGLBackend.create({ canvas, dtype })
}

async function fetchBin(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url)
  if (!r.ok || (r.headers.get('content-type') ?? '').startsWith('text/html')) return null
  return r.arrayBuffer()
}
async function fetchWeights(tier: string, dtype: Dtype): Promise<ArrayBuffer> {
  const base = `/model_${tier}_flow`
  if (dtype === 'f16') { const f16 = await fetchBin(`${base}.f16.bin`); if (f16) return f16 }
  const f32 = await fetchBin(`${base}.bin`)
  if (!f32) throw new Error(`failed to fetch ${base}.bin`)
  return f32
}

type Source = { grab: () => Promise<ImageBitmap>; stop: () => void }

async function loadSource(kind: string): Promise<Source> {
  if (kind === 'image') {
    const blob = await (await fetch('/test_img.jpg')).blob()
    const bm = await createImageBitmap(blob)
    return { grab: () => createImageBitmap(bm), stop: () => bm.close() }
  }
  const v = document.createElement('video')
  v.muted = true; v.playsInline = true
  if (kind === 'video') { v.src = '/loop_video.mp4'; v.loop = true }
  else v.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false })
  await v.play()
  return {
    grab: () => createImageBitmap(v),
    stop: () => {
      v.pause()
      if (v.srcObject) for (const t of (v.srcObject as MediaStream).getTracks()) t.stop()
    },
  }
}

let session = 0
let stopSource: (() => void) | null = null
let requestManual = false
$('reframe-now').addEventListener('click', () => { requestManual = true })

async function run() {
  const mySession = ++session
  await new Promise(r => requestAnimationFrame(r))
  stopSource?.(); stopSource = null
  try {
    const tier = val('tier'), backendName = val('backend'), dtype = val('dtype') as Dtype
    const cfg = TIER_CONFIG[tier]
    const c = cfg.canvasRes

    status(`loading ${tier}…`)
    const [wbuf, source] = await Promise.all([fetchWeights(tier, dtype), loadSource(val('source'))])
    stopSource = source.stop
    const backend = await createBackend(backendName, dtype, new OffscreenCanvas(4, 4))
    const w = loadWeightsFromBinary(wbuf) as any
    if (!w.face) throw new Error('weights have no face blob')

    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const face = new FaceHeatmapNet(backend, matting.encoderTaps, w.face)
    let boxOp: Op = backend.ops.FaceBoxesFromHeatmaps(face.output, {
      win: 3, thresh: 0.15, boxScale: 2.4, maxFaces: 4, tol: 0.6,
    })
    // Alpha (channel 0 of the tier output, canvas res) → tiny grid for the body
    // centroid. Only read back when the body actually has weight — on WebGL the
    // readback is a blocking readPixels (see project_webgl_readback_blocks).
    const maskDown = backend.ops.BilinearUpsample(matting.output, { outH: MASK_H, outW: MASK_W })

    const view = $<HTMLCanvasElement>('view'); view.width = c.w; view.height = c.h
    const vctx = view.getContext('2d')!
    const out = $<HTMLCanvasElement>('out'); out.width = c.w; out.height = c.h
    const octx = out.getContext('2d')!

    const smoother = new Smoother()
    let frozen: Rect | null = null
    let emaFps = 0, last = performance.now()
    let lastMode = val('mode')

    status(`running ${tier} — reframe v0 (CPU)`)
    const tick = async () => {
      if (session !== mySession) return
      let bm: ImageBitmap
      try { bm = await source.grab() } catch { return }
      if (session !== mySession) { bm.close(); return }

      netInput.setSource(bm)
      netInput.run()
      matting.run()
      face.run()
      boxOp.run()
      const bx = await backend.readback(boxOp.output)

      const mode = val('mode')
      if (mode !== lastMode) { smoother.reset(); frozen = null; lastMode = mode }

      // Subject = largest live face. (Production wants hysteresis here so a
      // background face that briefly measures larger can't steal the frame.)
      const boxes: Box[] = []
      for (let i = 0; i < 4; i++) {
        const s = bx[i * 4 + 3]
        if (s > 0) boxes.push({ cx: bx[i * 4], cy: bx[i * 4 + 1], hs: bx[i * 4 + 2], score: s })
      }
      const subject = boxes.slice().sort((a, b) => b.hs - a.hs)[0] ?? null

      const bodyW = numv('bodyw')
      let body: { x: number; y: number } | null = null
      if (bodyW > 0) {
        maskDown.run()
        const m = await backend.readback(maskDown.output)
        let sx = 0, sy = 0, sw = 0
        for (let y = 0; y < MASK_H; y++) for (let x = 0; x < MASK_W; x++) {
          const a = m[(y * MASK_W + x) * 4]      // alpha in channel 0
          if (a > 0.5) { sx += (x + 0.5) * a; sy += (y + 0.5) * a; sw += a }
        }
        if (sw > 1) body = { x: sx / sw / MASK_W, y: sy / sw / MASK_H }
      }

      // ── the reframe rule ────────────────────────────────────────────────
      let want: Rect | null = null
      if (subject) {
        const target = body
          ? { x: lerp(subject.cx, body.x, bodyW), y: lerp(subject.cy, body.y, bodyW) }
          : { x: subject.cx, y: subject.cy }
        want = solveCrop(
          target, subject.hs, subject.hs * c.w / c.h,
          numv('zoom'), numv('gravity'), numv('margin'),
        )
      }

      let rect: Rect | null
      if (mode === 'auto') {
        rect = want ? smoother.step(want, numv('dead'), numv('ease')) : smoother.cur
      } else {
        if (requestManual && want) frozen = { ...want }   // solve once, then freeze
        rect = frozen
      }
      requestManual = false

      // ── draw ────────────────────────────────────────────────────────────
      vctx.drawImage(bm, 0, 0, view.width, view.height)
      if ($<HTMLInputElement>('debug').checked) {
        for (const b of boxes) {
          vctx.strokeStyle = b === subject ? '#2a6' : '#666'
          vctx.lineWidth = b === subject ? 2 : 1
          const hx = b.hs * view.width, hy = b.hs * view.width
          vctx.strokeRect(b.cx * view.width - hx, b.cy * view.height - hy, hx * 2, hy * 2)
        }
        vctx.fillStyle = '#fff'
        vctx.fillRect(view.width / 2 - 2, view.height / 2 - 2, 4, 4)   // frame centre = the base
        if (body) {
          vctx.fillStyle = '#5fb8ff'
          vctx.beginPath(); vctx.arc(body.x * view.width, body.y * view.height, 5, 0, Math.PI * 2); vctx.fill()
        }
      }
      if (rect) {
        vctx.strokeStyle = '#ffd23f'; vctx.lineWidth = 2
        vctx.strokeRect((rect.cx - rect.w / 2) * view.width, (rect.cy - rect.h / 2) * view.height,
                        rect.w * view.width, rect.h * view.height)
        octx.drawImage(bm,
          (rect.cx - rect.w / 2) * bm.width, (rect.cy - rect.h / 2) * bm.height,
          rect.w * bm.width, rect.h * bm.height, 0, 0, out.width, out.height)
      } else {
        octx.drawImage(bm, 0, 0, out.width, out.height)   // no-op → the raw frame
      }
      bm.close()

      const now = performance.now()
      emaFps = emaFps ? emaFps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last)
      last = now
      const zNow = rect ? (1 / rect.w).toFixed(2) : '—'
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · ${mode}`
        + ` · ${boxes.length} face(s) · zoom ${zNow}`
        + ` · ${want ? (smoother.isMoving && mode === 'auto' ? 'MOVING' : 'held') : 'NO-OP (cannot frame subject)'}`)
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['zoom', 'gravity', 'bodyw', 'margin', 'dead', 'ease']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()
$('run').addEventListener('click', () => run())
