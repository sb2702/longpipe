// Source orientation: the display transform a frame still needs when its
// pixels are NOT upright (camera sensor orientation) and the browser's upload
// path won't apply it for us.
//
// Semantics follow WebCodecs VideoFrame.rotation / flip: to DISPLAY the frame,
// rotate it `rotation` degrees clockwise, THEN flip horizontally if `flip`
// ("Flip applies after rotation", https://w3c.github.io/webcodecs/#rendering).
//
// Who sets it: browsers that implement VideoFrame.rotation (Chrome 138+) apply
// it themselves in importExternalTexture / copyExternalImageToTexture /
// createImageBitmap / texImage2D, so for them the residual is the identity.
// Shipping Safari tags camera frames with a hidden rotation it does NOT expose
// and every upload path ignores (WebKit WebCodecsVideoFrameRotationEnabled is
// off), so the main thread measures it (see pipeline/orientation_probe.ts) and
// the Input op applies it while sampling (no extra copy).

export type Rotation = 0 | 90 | 180 | 270

export interface Orientation {
  rotation: Rotation
  flip:     boolean
}

export const IDENTITY_ORIENTATION: Orientation = Object.freeze({ rotation: 0, flip: false }) as Orientation

export function isIdentity(o: Orientation | null | undefined): boolean {
  return !o || (o.rotation === 0 && !o.flip)
}

export function sameOrientation(a: Orientation | null | undefined, b: Orientation | null | undefined): boolean {
  const x = a ?? IDENTITY_ORIENTATION, y = b ?? IDENTITY_ORIENTATION
  return x.rotation === y.rotation && x.flip === y.flip
}

export function normalizeRotation(r: unknown): Rotation {
  const n = ((Math.round(Number(r) / 90) * 90) % 360 + 360) % 360
  return (Number.isFinite(n) ? n : 0) as Rotation
}

// Upright (displayed) size of a w×h source that needs `o`.
export function uprightSize(w: number, h: number, o: Orientation | null | undefined): { w: number; h: number } {
  return o && (o.rotation === 90 || o.rotation === 270) ? { w: h, h: w } : { w, h }
}

// Affine map from an UPRIGHT (output) uv in [0,1]² to the SOURCE uv to sample:
//   su = m[0]*u + m[1]*v + m[2]
//   sv = m[3]*u + m[4]*v + m[5]
// Derivation: display = flipH(rotCW(source)). Inverse: undo the flip
// (u → 1-u), then undo the clockwise rotation:
//   90:  (u,v) → (v, 1-u)     180: (u,v) → (1-u, 1-v)     270: (u,v) → (1-v, u)
export type UvMatrix = [number, number, number, number, number, number]

export function orientationUvMatrix(o: Orientation | null | undefined): UvMatrix {
  const rot = o?.rotation ?? 0
  // Rotation inverse as a matrix on (u, v, 1).
  let m: UvMatrix
  switch (rot) {
    case 90:  m = [0, 1, 0, -1, 0, 1]; break
    case 180: m = [-1, 0, 1, 0, -1, 1]; break
    case 270: m = [0, -1, 1, 1, 0, 0]; break
    default:  m = [1, 0, 0, 0, 1, 0]
  }
  if (o?.flip) {
    // Compose with u → 1-u applied first: replace u by (1-u).
    m = [-m[0], m[1], m[0] + m[2], -m[3], m[4], m[3] + m[5]]
  }
  return m
}

export function applyUvMatrix(m: UvMatrix, u: number, v: number): [number, number] {
  return [m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5]]
}

// Per-frame orientation side channel inside the worker: the postMessage input
// adapter records what main measured for each frame; the renderer looks it up.
// A WeakMap so frames that are dropped/closed don't need bookkeeping.
const frameOrientation = new WeakMap<object, Orientation>()

export function tagFrameOrientation(frame: object, o: Orientation | null | undefined): void {
  if (o && !isIdentity(o)) frameOrientation.set(frame, o)
}

export function frameResidualOrientation(frame: object): Orientation {
  return frameOrientation.get(frame) ?? IDENTITY_ORIENTATION
}
