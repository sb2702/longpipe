// Background option parsing — main-thread boundary between the wide public
// input surface and the narrow canonical form the worker consumes.
//
// All async loading (URL fetch, Blob → ImageBitmap decode) happens here so
// the worker only ever sees fully-resolved data. Pipeline.bootstrap()
// awaits normalizeBackground() before sending startRender to the worker.

const SIGMA_LOW    = 4
const SIGMA_MEDIUM = 8
const SIGMA_HIGH   = 16

// ---- Public input types ----

export type BackgroundInput =
  | 'blur' | 'none'
  | string                                 // URL → image
  | ImageBitmap
  | HTMLImageElement
  | HTMLVideoElement                       // typed but throws — see normalize
  | { color: [number, number, number] }    // debug, undocumented
  | { blur:  BlurInput }
  | { image: ImageInput }
  | { video: VideoInput }                  // typed but throws — see normalize

export type BlurInput =
  | true                                   // default (medium)
  | { strength: 'low' | 'medium' | 'high' | number }   // number is 0..1
  | { sigma: number }                                  // raw escape hatch

export type ImageInput =
  | ImageBitmap
  | HTMLImageElement
  | string                                 // URL
  | { data: ArrayBuffer | Blob | Uint8Array; type: string }

export type VideoInput =
  | HTMLVideoElement
  | string
  | { data: ArrayBuffer | Blob | Uint8Array; type: string }

// ---- Canonical (post-parse) type — what the worker sees ----

export type Background =
  | { kind: 'none' }
  | { kind: 'color'; rgb: [number, number, number] }
  | { kind: 'blur';  sigma: number }
  | { kind: 'image'; bitmap: ImageBitmap }

// ---- Parser ----

export async function normalizeBackground(input: BackgroundInput): Promise<Background> {
  if (typeof input === 'string') {
    if (input === 'none') return { kind: 'none' }
    if (input === 'blur') return { kind: 'blur', sigma: SIGMA_MEDIUM }
    return { kind: 'image', bitmap: await loadImageFromUrl(input) }
  }

  if (input instanceof ImageBitmap) {
    return { kind: 'image', bitmap: input }
  }
  if (typeof HTMLImageElement !== 'undefined' && input instanceof HTMLImageElement) {
    return { kind: 'image', bitmap: await createImageBitmap(input) }
  }
  if (typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement) {
    throw new Error('background: video element input is not yet implemented')
  }

  // Cast to the object-form union: by this point we've ruled out strings
  // and DOM elements, but TS can't narrow because the typeof guards above
  // are conservative (HTMLImageElement etc. have `.blur()` methods that
  // confuse `'blur' in input`).
  const obj = input as { color?: [number, number, number]; blur?: BlurInput; image?: ImageInput; video?: VideoInput }
  if (obj.color) return { kind: 'color', rgb: obj.color }
  if (obj.blur  !== undefined) return { kind: 'blur',  sigma: parseBlur(obj.blur) }
  if (obj.image !== undefined) return { kind: 'image', bitmap: await parseImage(obj.image) }
  if (obj.video !== undefined) throw new Error('background: video input is not yet implemented')

  throw new Error(`background: unrecognized input shape — ${describe(input)}`)
}

function parseBlur(b: BlurInput): number {
  if (b === true) return SIGMA_MEDIUM
  if ('sigma' in b) {
    if (typeof b.sigma !== 'number' || !Number.isFinite(b.sigma) || b.sigma < 0)
      throw new Error(`background.blur.sigma must be a non-negative number, got ${b.sigma}`)
    return b.sigma
  }
  if ('strength' in b) {
    const s = b.strength
    if (typeof s === 'number') {
      const t = Math.max(0, Math.min(1, s))
      return SIGMA_LOW + (SIGMA_HIGH - SIGMA_LOW) * t
    }
    if (s === 'low')    return SIGMA_LOW
    if (s === 'medium') return SIGMA_MEDIUM
    if (s === 'high')   return SIGMA_HIGH
    throw new Error(`background.blur.strength must be 'low' | 'medium' | 'high' | number, got ${describe(s)}`)
  }
  throw new Error(`background.blur: must be true, { strength }, or { sigma } — got ${describe(b)}`)
}

async function parseImage(i: ImageInput): Promise<ImageBitmap> {
  if (i instanceof ImageBitmap) return i
  if (typeof HTMLImageElement !== 'undefined' && i instanceof HTMLImageElement) {
    return createImageBitmap(i)
  }
  if (typeof i === 'string') return loadImageFromUrl(i)
  if (i && typeof i === 'object' && 'data' in i && 'type' in i) {
    const blob = i.data instanceof Blob
      ? i.data
      : new Blob([i.data as BlobPart], { type: i.type })
    return createImageBitmap(blob)
  }
  throw new Error(`background.image: unrecognized shape — ${describe(i)}`)
}

async function loadImageFromUrl(url: string): Promise<ImageBitmap> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`background: failed to load image from ${url} (HTTP ${res.status})`)
  const blob = await res.blob()
  return createImageBitmap(blob)
}

function describe(v: unknown): string {
  if (v === null) return 'null'
  if (typeof v !== 'object') return typeof v + '(' + String(v) + ')'
  const ctor = (v as object).constructor?.name ?? 'Object'
  return `${ctor} ${JSON.stringify(v).slice(0, 80)}`
}
