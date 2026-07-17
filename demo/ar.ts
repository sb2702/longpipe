import type { Backend, Dtype } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { TierModel } from '~/model/tier_model'
import { TIER_CONFIG } from '~/model/tier_config'
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net'
import { LandmarkNet } from '~/model/networks/landmark_net'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { ArScene, type Pose as GlPose, type Mesh } from './ar_gl'

// ─────────────────────────────────────────────────────────────────────────────
// Face AR PoC — dummy 3D sunglasses. Demo-only; the point is to answer one
// question: do our landmarks carry enough signal to hang AR geometry off?
//
// The POSE FIT is on the CPU (cheap, and easy to iterate on). The RENDERING is
// real WebGL2 with a depth buffer, in its own context (ar_gl.ts) — nothing is
// shared with the SDK backend, because the landmarks come back to the CPU anyway
// and the video is uploaded to the AR context directly.
//
// It has a depth-only face OCCLUDER, which is the trick jeeliz's glassesVTO uses
// (their `occluderURL: models3D/face.json`) and the difference between glasses on
// a face and glasses pasted on a photo: the far temple gets depth-tested away
// instead of tracking across the cheek. We build ours from the canonical mesh —
// face_topology.json's `idx` is the 898-triangle list, as landmark indices.
//
// NOTHING about the landmark model changes. LandmarkNet stays 2D by design —
// MediaPipe's z is unreliable and we never wanted it. The depth here comes from
// the CANONICAL FACE MODEL: one fixed, known 3D face (canonical_face.json,
// exported from ar-scope/canonical_face_model.obj). That's the standard AR fit —
// known 3D model points + observed 2D points → pose — and it's exactly why the
// model's own z output being useless doesn't matter: it's never consulted.
//
// The fit is POSIT — iterated weak perspective with a depth correction, i.e.
// TRUE perspective. It replaced a plain scaled-orthographic fit that was
// measurably wrong: with PERFECT landmarks, 6° of pose error head-on and 14° at
// 45° yaw. POSIT measures 0.00° across that sweep, 0.29° at 3px landmark noise.
//
// It keeps the property that makes the maths cheap: **AᵀA depends only on the
// (centred) canonical model** — constant, inverted once at init. Each iteration
// is Aᵀb (a sum over landmarks) + a 4×4 matvec: a reduction and a tiny solve,
// i.e. one workgroup, the same shape as face_box / reframe_state, if this ever
// graduates to the GPU.
// ─────────────────────────────────────────────────────────────────────────────

const CROP = 256
const N_PTS = 478
const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406]
const IMAGENET_STD:  [number, number, number] = [0.229, 0.224, 0.225]
// Rigid subset for the pose fit, picked GEOMETRICALLY (canonical y above this)
// rather than from a hand-copied index list, so the mouth can't drag the pose
// around when you talk. Everything above the jaw. y > 1.0 (the old value) excluded the
// ears — the only points with real depth — leaving a near-planar band. POSIT
// fits a plane fine, but depth spread costs nothing and helps under noise.
const RIGID_MIN_Y = -4.0
const POSIT_ITERS = 8   // converges in ~4; 8 is free at this size

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const status = (s: string) => { $('status').textContent = s }
const numv = (id: string) => parseFloat($<HTMLInputElement>(id).value)
const val = (id: string) => $<HTMLSelectElement>(id).value
const on = (id: string) => $<HTMLInputElement>(id).checked

type V3 = [number, number, number]

// ── linear algebra (4×4 solve, once per frame per axis) ─────────────────────

function invert4(m: number[][]): number[][] {
  const a = m.map((r, i) => [...r, ...[0, 0, 0, 0].map((_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < 4; c++) {
    let p = c
    for (let r = c + 1; r < 4; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r
    if (Math.abs(a[p][c]) < 1e-12) throw new Error('canonical AᵀA is singular')
    ;[a[c], a[p]] = [a[p], a[c]]
    const d = a[c][c]
    for (let j = 0; j < 8; j++) a[c][j] /= d
    for (let r = 0; r < 4; r++) {
      if (r === c) continue
      const f = a[r][c]
      if (!f) continue
      for (let j = 0; j < 8; j++) a[r][j] -= f * a[c][j]
    }
  }
  return a.map(r => r.slice(4))
}

const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const norm3 = (a: V3) => Math.hypot(a[0], a[1], a[2])
const cross3 = (a: V3, b: V3): V3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

interface Pose { R0: V3; R1: V3; R2: V3; T: V3; f: number; pp: [number, number]; residual: number }

// ── the demo ────────────────────────────────────────────────────────────────

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
async function fetchWeights(base: string, dtype: Dtype): Promise<ArrayBuffer> {
  if (dtype === 'f16') { const f = await fetchBin(`${base}.f16.bin`); if (f) return f }
  const f32 = await fetchBin(`${base}.bin`)
  if (!f32) throw new Error(`failed to fetch ${base}.bin`)
  return f32
}

type Source = { grab: () => Promise<ImageBitmap>; stop: () => void }

async function loadSource(kind: string): Promise<Source> {
  if (kind === 'image') {
    const bm = await createImageBitmap(await (await fetch('/test_img.jpg')).blob())
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

type Tri = { pos: number[]; nrm: number[] }

const push3 = (t: Tri, p: V3, n: V3) => { t.pos.push(p[0], p[1], p[2]); t.nrm.push(n[0], n[1], n[2]) }
const quad = (t: Tri, a: V3, b: V3, c: V3, d: V3, n: V3) => {
  push3(t, a, n); push3(t, b, n); push3(t, c, n)
  push3(t, a, n); push3(t, c, n); push3(t, d, n)
}

// three.js BufferGeometry v3 loader (jeeliz's models3D/*.json). Flat position +
// index arrays — the easy format, no three.js needed. They carry NO normals, so
// we derive flat ones per triangle after de-indexing.
function loadBufferGeometry(json: any, xf: (p: V3) => V3): Tri {
  const d = json.data
  const P: number[] = d.attributes.position.array
  const I: number[] = d.index ? d.index.array : Array.from({ length: P.length / 3 }, (_, i) => i)
  const t: Tri = { pos: [], nrm: [] }
  const at = (k: number): V3 => xf([P[k * 3], P[k * 3 + 1], P[k * 3 + 2]])
  for (let i = 0; i < I.length; i += 3) {
    const a = at(I[i]), b = at(I[i + 1]), c = at(I[i + 2])
    const u: V3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    const w: V3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    const n = cross3(u, w)
    const l = norm3(n) || 1
    const nn: V3 = [n[0] / l, n[1] / l, n[2] / l]
    push3(t, a, nn); push3(t, b, nn); push3(t, c, nn)
  }
  return t
}

// Map jeeliz's model frame into OUR canonical face space. Their meshes share one
// frame with the same axis convention as ours (+x right, +y up, +z forward), so
// ONE similarity transform derived from the lenses places all of them — frames,
// lenses, and their head occluder alike.
//
// Anchored on geometry, not guesswork: scale so the lens x-span matches our
// ear-to-ear width, then drop the lens centre onto our eye midpoint pushed
// forward. The sliders nudge from there.
function jeelizTransform(lensesJson: any, C: Float32Array, fwd: number, sizeMul: number): (p: V3) => V3 {
  const P: number[] = lensesJson.data.attributes.position.array
  let lo: V3 = [1e9, 1e9, 1e9], hi: V3 = [-1e9, -1e9, -1e9]
  for (let i = 0; i < P.length; i += 3)
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]) }
  const theirW = hi[0] - lo[0]
  const theirC: V3 = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]

  const Q = (i: number): V3 => [C[i * 3], C[i * 3 + 1], C[i * 3 + 2]]
  const ourW = Math.abs(Q(454)[0] - Q(234)[0]) * sizeMul      // ear to ear
  const s = ourW / theirW
  const eyeL = Q(33), eyeLi = Q(133), eyeR = Q(263), eyeRi = Q(362)
  const eyeMidY = (eyeL[1] + eyeLi[1] + eyeR[1] + eyeRi[1]) / 4
  const eyeMidZ = (eyeL[2] + eyeLi[2] + eyeR[2] + eyeRi[2]) / 4
  const t: V3 = [-s * theirC[0], eyeMidY - s * theirC[1], eyeMidZ + fwd - s * theirC[2]]
  return (p: V3): V3 => [s * p[0] + t[0], s * p[1] + t[1], s * p[2] + t[2]]
}

// Depth-only face occluder from the canonical mesh. face_topology.json's `idx`
// is a flat list of LANDMARK indices, three per triangle — so the canonical 3D
// positions turn straight into a face mesh with no extra asset.
function buildOccluder(C: Float32Array, idx: Float32Array): Tri {
  const t: Tri = { pos: [], nrm: [] }
  const P = (i: number): V3 => [C[i * 3], C[i * 3 + 1], C[i * 3 + 2]]
  for (let v = 0; v < idx.length; v += 3) {
    const a = P(idx[v]), b = P(idx[v + 1]), c = P(idx[v + 2])
    const u: V3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    const w: V3 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    const n = cross3(u, w)
    const l = norm3(n) || 1
    const nn: V3 = [n[0] / l, n[1] / l, n[2] / l]
    push3(t, a, nn); push3(t, b, nn); push3(t, c, nn)
  }
  return t
}

// Procedural sunglasses, built in CANONICAL model space off real landmark
// positions — no magic numbers, no asset to license. Two lens discs, a bridge,
// two temples running back to the ear points.
function glassesGeometry(C: Float32Array, fwd: number, sizeMul: number) {
  const P = (i: number): V3 => [C[i * 3], C[i * 3 + 1], C[i * 3 + 2]]
  const mid = (a: V3, b: V3): V3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
  const FWD: V3 = [0, 0, 1]

  const lensOf = (outer: number, inner: number) => {
    const o = P(outer), n = P(inner)
    const c = mid(o, n)
    const r = norm3([o[0] - n[0], o[1] - n[1], o[2] - n[2]] as V3) / 2
    return { c: [c[0], c[1], c[2] + fwd] as V3, rx: r * 1.45 * sizeMul, ry: r * 1.05 * sizeMul }
  }
  const L = lensOf(33, 133)     // model −x side
  const R = lensOf(263, 362)    // model +x side
  const N = 32

  // Lens disc: a triangle fan facing +z (forward, toward the camera at rest).
  const lensTri = (l: ReturnType<typeof lensOf>): Tri => {
    const t: Tri = { pos: [], nrm: [] }
    const at = (k: number): V3 => {
      const a = (k / N) * Math.PI * 2
      return [l.c[0] + l.rx * Math.cos(a), l.c[1] + l.ry * Math.sin(a), l.c[2]]
    }
    for (let k = 0; k < N; k++) { push3(t, l.c, FWD); push3(t, at(k), FWD); push3(t, at(k + 1), FWD) }
    return t
  }

  // Frame: rim tubes around each lens, the bridge, and the temples.
  const frame: Tri = { pos: [], nrm: [] }
  const rim = (l: ReturnType<typeof lensOf>) => {
    const th = 0.16 * sizeMul, d = 0.13 * sizeMul
    const at = (k: number, ex: number): V3 => {
      const a = (k / N) * Math.PI * 2
      return [l.c[0] + (l.rx + ex) * Math.cos(a), l.c[1] + (l.ry + ex) * Math.sin(a), l.c[2]]
    }
    for (let k = 0; k < N; k++) {
      const a = (k / N) * Math.PI * 2
      const nrm: V3 = [Math.cos(a), Math.sin(a), 0]
      const i0 = at(k, 0), i1 = at(k + 1, 0), o0 = at(k, th), o1 = at(k + 1, th)
      quad(frame, i0, i1, o1, o0, FWD)                                   // front face
      quad(frame, [o0[0], o0[1], o0[2] - d], [o1[0], o1[1], o1[2] - d], o1, o0, nrm)  // outer wall
    }
  }
  rim(L); rim(R)

  const by = (L.c[1] + R.c[1]) / 2, bz = (L.c[2] + R.c[2]) / 2, bh = 0.2 * sizeMul
  quad(frame,
    [L.c[0] + L.rx * 0.9, by + bh, bz], [R.c[0] - R.rx * 0.9, by + bh, bz],
    [R.c[0] - R.rx * 0.9, by - bh, bz], [L.c[0] + L.rx * 0.9, by - bh, bz], FWD)

  // Temples: lens outer edge back to the ear point. These are what make yaw
  // legible — and what the occluder has to hide on the far side.
  const temple = (l: ReturnType<typeof lensOf>, ear: number, sign: number) => {
    const e = P(ear)
    const a: V3 = [l.c[0] + sign * (l.rx + 0.1), l.c[1] + 0.1, l.c[2] - 0.1]
    const b: V3 = [e[0] + sign * 0.2, e[1] + 0.7, e[2] + 0.3]
    const h = 0.18 * sizeMul
    const nrm: V3 = [sign, 0, 0]
    quad(frame, [a[0], a[1] + h, a[2]], [b[0], b[1] + h, b[2]], [b[0], b[1] - h, b[2]], [a[0], a[1] - h, a[2]], nrm)
  }
  temple(L, 234, -1)
  temple(R, 454, +1)

  return { frame, lensL: lensTri(L), lensR: lensTri(R) }
}

let session = 0
let stopSource: (() => void) | null = null

async function run() {
  const mySession = ++session
  await new Promise(r => requestAnimationFrame(r))
  stopSource?.(); stopSource = null
  try {
    const tier = val('tier'), backendName = val('backend'), dtype = val('dtype') as Dtype
    const cfg = TIER_CONFIG[tier]
    const c = cfg.canvasRes

    status(`loading ${tier} + landmark + canonical face…`)
    const [tierBuf, lmBuf, canonJson, topoJson, jzFrames, jzLenses, jzFace, source] = await Promise.all([
      fetchWeights(`/model_${tier}_flow`, dtype),
      fetchWeights('/model_landmark_mesh', dtype),
      fetch('/canonical_face.json').then(r => r.json()),
      fetch('/face_topology.json').then(r => r.json()),
      // jeeliz's glassesVTO models — pulled in verbatim so the comparison against
      // their demo is the same mesh on our tracking, not our art vs theirs.
      fetch('/jeeliz/glassesFramesBranchesBent.json').then(r => r.json()),
      fetch('/jeeliz/glassesLenses.json').then(r => r.json()),
      fetch('/jeeliz/face.json').then(r => r.json()),
      loadSource(val('source')),
    ])
    stopSource = source.stop
    const backend = await createBackend(backendName, dtype, new OffscreenCanvas(4, 4))
    const w = loadWeightsFromBinary(tierBuf) as any
    if (!w.face) throw new Error('tier weights have no face blob')

    const CANON = new Float32Array(canonJson.pos)
    const nCanon = canonJson.count as number

    // Rigid subset + the CONSTANT normal matrix. POSIT works on the model CENTRED
    // on a reference point, so AᵀA is built from the centred points — still purely
    // canonical, so it never changes and is inverted once, here.
    const rigid: number[] = []
    for (let i = 0; i < nCanon; i++) if (CANON[i * 3 + 1] > RIGID_MIN_Y) rigid.push(i)
    const cen: V3 = [0, 0, 0]
    for (const i of rigid) for (let a = 0; a < 3; a++) cen[a] += CANON[i * 3 + a] / rigid.length
    const AtA = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for (const i of rigid) {
      const row = [CANON[i * 3] - cen[0], CANON[i * 3 + 1] - cen[1], CANON[i * 3 + 2] - cen[2], 1]
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) AtA[a][b] += row[a] * row[b]
    }
    const AtAinv = invert4(AtA)

    const netInput = backend.ops.Input(c.h, c.w)
    const matting = new TierModel(backend, netInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base)
    const face = new FaceHeatmapNet(backend, matting.encoderTaps, w.face)
    const boxOp = backend.ops.FaceBoxFromHeatmaps(face.output, { win: 3, thresh: 0.15, boxScale: 2.4 })
    const crop = backend.ops.CropResample(netInput.output, boxOp.output, {
      outH: CROP, outW: CROP, mean: IMAGENET_MEAN, std: IMAGENET_STD,
    })
    const lm = new LandmarkNet(backend, crop.output, loadWeightsFromBinary(lmBuf) as any)

    // Fresh canvas per run — a canvas is permanently bound to its first context
    // type, so re-running (or switching backend) needs a new element.
    const oldView = $<HTMLCanvasElement>('view')
    const view = document.createElement('canvas')
    view.id = 'view'
    oldView.replaceWith(view)
    view.width = c.w; view.height = c.h
    const scene = new ArScene(view)

    const mk = (t: Tri) => scene.makeMesh(new Float32Array(t.pos), new Float32Array(t.nrm))

    // Our canonical-mesh occluder: a FACE, so it stops around the ears. Good
    // enough to hide a temple crossing the cheek, but a temple continuing past
    // the silhouette has nothing behind it. jeeliz's face.json is a full HEAD
    // (279 tall × 186 wide × 244 deep), so it occludes properly — and because
    // their meshes share one frame, the same transform places it.
    const occCanonical = mk(buildOccluder(CANON, new Float32Array(topoJson.idx)))

    // Rebuilt on any slider/selector change: the alignment depends on fwd/size.
    let glasses: { frame: Mesh; lensL: Mesh; lensR: Mesh } | null = null
    let jeeliz: { frames: Mesh; lenses: Mesh; head: Mesh } | null = null
    const rebuild = () => {
      const fwd = numv('fwd'), size = numv('size')
      if (val('model') === 'dummy') {
        const g = glassesGeometry(CANON, fwd, size)
        glasses = { frame: mk(g.frame), lensL: mk(g.lensL), lensR: mk(g.lensR) }
        jeeliz = null
      } else {
        const xf = jeelizTransform(jzLenses, CANON, fwd, size)
        jeeliz = {
          frames: mk(loadBufferGeometry(jzFrames, xf)),
          lenses: mk(loadBufferGeometry(jzLenses, xf)),
          head:   mk(loadBufferGeometry(jzFace, xf)),
        }
        glasses = null
      }
    }
    rebuild()
    for (const id of [...SLIDERS, 'model']) $(id).addEventListener('input', rebuild)
    $('model').addEventListener('change', rebuild)

    const obs = new Float64Array(N_PTS * 2)   // landmarks in canvas px

    let emaFps = 0, last = performance.now()
    status(`running ${tier} — ${rigid.length}-point rigid fit`)

    const tick = async () => {
      if (session !== mySession) return
      let bm: ImageBitmap
      try { bm = await source.grab() } catch { return }
      if (session !== mySession) { bm.close(); return }

      netInput.setSource(bm)
      netInput.run(); matting.run(); face.run(); boxOp.run(); crop.run(); lm.run()
      const [box, lmv] = await Promise.all([backend.readback(boxOp.output), backend.readback(lm.output)])

      const score = box[3]
      let pose: Pose | null = null
      if (score > 0) {
        // Landmarks are [0,1] CROP coords → frame fractions → canvas px. halfSide
        // is a fraction of WIDTH; the y half-extent rescales by the aspect (the
        // box is square in pixels). Same mapping as the overlay shaders.
        const hsx = box[2], hsy = box[2] * c.w / c.h
        for (let i = 0; i < N_PTS; i++) {
          const lx = lmv[i * 2], ly = lmv[i * 2 + 1]
          obs[i * 2]     = ((box[0] - hsx) + lx * 2 * hsx) * view.width
          obs[i * 2 + 1] = ((box[1] - hsy) + ly * 2 * hsy) * view.height
        }

        // POSIT — iterated weak perspective with a depth correction. Two things
        // matter here and both were bugs the first time round:
        //
        //  1. The fit works in a STANDARD camera frame (u = f·X/Z, v = f·Y/Z).
        //     Canvas y is down, so v is negated on the way IN and flipped back on
        //     the way out (in the shader). Feeding y-down straight into POSIT makes
        //     e3 = −R[2], so the ε depth correction pushes the WRONG WAY and the
        //     result is worse than not correcting at all.
        //  2. The reference image point u₀ is an UNKNOWN — it's the '1' column of
        //     the design matrix. Drop it and ε is anchored to nothing.
        //
        // AᵀA is built only from the (constant) centred model, so it's inverted
        // once at init; each iteration is a reduction + a 4×4 matvec.
        const f = numv('focal') * view.width
        const ppx = view.width / 2, ppy = view.height / 2
        const eps = new Float64Array(rigid.length)
        let R0: V3 = [1, 0, 0], R1: V3 = [0, 1, 0], R2: V3 = [0, 0, 1]
        let Z0 = f, u0 = 0, v0 = 0
        for (let it = 0; it < POSIT_ITERS; it++) {
          const bx = [0, 0, 0, 0], by = [0, 0, 0, 0]
          for (let k = 0; k < rigid.length; k++) {
            const i = rigid[k]
            const row = [CANON[i * 3] - cen[0], CANON[i * 3 + 1] - cen[1], CANON[i * 3 + 2] - cen[2], 1]
            const uu = (obs[i * 2] - ppx) * (1 + eps[k])
            const vv = -(obs[i * 2 + 1] - ppy) * (1 + eps[k])   // canvas y down → camera y up
            for (let a = 0; a < 4; a++) { bx[a] += row[a] * uu; by[a] += row[a] * vv }
          }
          const solve = (b: number[]) => AtAinv.map(r => r[0] * b[0] + r[1] * b[1] + r[2] * b[2] + r[3] * b[3])
          const px = solve(bx), py = solve(by)
          const I: V3 = [px[0], px[1], px[2]], J: V3 = [py[0], py[1], py[2]]
          u0 = px[3]; v0 = py[3]
          const nI = norm3(I) || 1e-9, nJ = norm3(J) || 1e-9
          Z0 = f / ((nI + nJ) / 2)
          R0 = [I[0] / nI, I[1] / nI, I[2] / nI]
          const pj = dot3(J, R0)
          const o: V3 = [J[0] - pj * R0[0], J[1] - pj * R0[1], J[2] - pj * R0[2]]
          const nO = norm3(o) || 1e-9
          R1 = [o[0] / nO, o[1] / nO, o[2] / nO]
          R2 = cross3(R0, R1)
          for (let k = 0; k < rigid.length; k++) {
            const i = rigid[k]
            const q: V3 = [CANON[i * 3] - cen[0], CANON[i * 3 + 1] - cen[1], CANON[i * 3 + 2] - cen[2]]
            eps[k] = dot3(q, R2) / Z0
          }
        }
        // u₀ = f·tx/Z₀ ⇒ the camera-space position of the model centre.
        const T: V3 = [u0 * Z0 / f, v0 * Z0 / f, Z0]

        // Residual: true reprojection error, in canvas px.
        let sse = 0
        for (let k = 0; k < rigid.length; k++) {
          const i = rigid[k]
          const q: V3 = [CANON[i * 3] - cen[0], CANON[i * 3 + 1] - cen[1], CANON[i * 3 + 2] - cen[2]]
          const zc = dot3(q, R2) + T[2]
          const ex = (ppx + f * (dot3(q, R0) + T[0]) / zc) - obs[i * 2]
          const ey = (ppy - f * (dot3(q, R1) + T[1]) / zc) - obs[i * 2 + 1]
          sse += ex * ex + ey * ey
        }
        pose = { R0, R1, R2, T, f, pp: [ppx, ppy], residual: Math.sqrt(sse / rigid.length) }
      }

      const glPose: GlPose | null = pose ? {
        r0: pose.R0, r1: pose.R1, r2: pose.R2,
        t: pose.T, center: cen, f: pose.f, pp: pose.pp,
      } : null

      const occMode = val('occluder')
      const occ = occMode === 'off' ? null
        : occMode === 'jeeliz' ? (jeeliz ? jeeliz.head : occCanonical)   // theirs only exists in jeeliz mode
        : occCanonical
      const parts = jeeliz
        ? [{ mesh: jeeliz.frames, color: [0.12, 0.13, 0.15, 1] as [number, number, number, number] }]
        : [{ mesh: glasses!.frame, color: [0.10, 0.12, 0.16, 1] as [number, number, number, number] }]
      const lenses = jeeliz
        ? [{ mesh: jeeliz.lenses, color: [0.05, 0.07, 0.10, 0.62] as [number, number, number, number] }]
        : [{ mesh: glasses!.lensL, color: [0.04, 0.06, 0.09, 0.68] as [number, number, number, number] },
           { mesh: glasses!.lensR, color: [0.04, 0.06, 0.09, 0.68] as [number, number, number, number] }]

      scene.render(bm, glPose, occ, parts, lenses)
      bm.close()

      const now = performance.now()
      emaFps = emaFps ? emaFps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last)
      last = now
      status(`${tier} ${backendName}/${dtype} · ${emaFps.toFixed(0)} fps · `
        + (pose
          ? `POSIT ${rigid.length} pts · residual ${pose.residual.toFixed(2)}px · `
            + `dist ${(pose.T[2] / pose.f * view.width).toFixed(0)}au · `
            + `yaw ${(Math.atan2(-pose.R2[0], pose.R2[2]) * 180 / Math.PI).toFixed(0)}° `
            + `pitch ${(Math.asin(Math.max(-1, Math.min(1, pose.R2[1]))) * 180 / Math.PI).toFixed(0)}°`
          : 'no face'))
      requestAnimationFrame(() => tick())
    }
    requestAnimationFrame(() => tick())
  } catch (err: any) {
    session++
    status(`error: ${err.message ?? err}`)
    console.error(err)
  }
}

const SLIDERS = ['fwd', 'size', 'focal']
const syncVals = () => { for (const id of SLIDERS) $(`${id}-v`).textContent = $<HTMLInputElement>(id).value }
for (const id of SLIDERS) $(id).addEventListener('input', syncVals)
syncVals()
$('run').addEventListener('click', () => run())
