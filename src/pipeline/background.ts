// Background option parsing — main-thread boundary between the wide public
// input surface and the narrow canonical form the worker consumes.
//
// All async loading (URL fetch, Blob → ImageBitmap decode, video element
// setup) happens here so the worker only ever sees fully-resolved data
// and ports it can use directly. Pipeline.bootstrap() awaits
// normalizeBackground() before sending startRender to the worker.
//
// For kinds that own resources (video element, MessagePort), the
// normalizer also returns a `transferList` (handed to postMessage so
// the port crosses zero-copy) and a `cleanup` callback (Pipeline calls
// it when this background is replaced or the pipeline is destroyed).

import { setupVideoSource, type VideoSourceInput } from './video_source'

const SIGMA_LOW    = 4
const SIGMA_MEDIUM = 8
const SIGMA_HIGH   = 16

// ---- Public input types ----

export type BackgroundInput =
  | 'blur' | 'none'
  | 'transparent'                          // isolate subject on transparency (matte → alpha)
  | 'matte'                                // render the raw alpha matte (white silhouette)
  | string                                 // URL → image
  | ImageBitmap
  | HTMLImageElement
  | HTMLVideoElement                       // typed but throws — see normalize
  | { color: ColorInput }
  | { blur:  BlurInput }
  | { image: ImageInput }
  | { video: VideoInput }                  // typed but throws — see normalize

export type BlurInput =
  | true                                   // default (medium)
  | { strength: 'low' | 'medium' | 'high' | number }   // number is 0..1
  | { sigma: number }                                  // raw escape hatch

export type ColorInput =
  | string                                 // hex: '#rgb' or '#rrggbb' (with or without leading #)
  | [number, number, number]               // [r, g, b], each float in [0, 1]

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
  | { kind: 'transparent' }                            // subject isolated on transparency
  | { kind: 'matte' }                                  // raw alpha matte (white silhouette)
  | { kind: 'color'; rgb: [number, number, number] }   // floats in [0, 1] — shader-ready
  | { kind: 'blur';  sigma: number }
  | { kind: 'image'; bitmap: ImageBitmap }
  | { kind: 'video'; port:   MessagePort }

// ---- Parser result wrapper ----
//
// `transferList` is included for kinds whose canonical form holds a
// transferable (the video MessagePort). `cleanup` is included when the
// normalizer constructed a resource that needs lifecycle management
// (the hidden video element + rVFC loop for video kind). Pipeline owns
// the cleanup and calls it on bg replacement / pipeline destroy.

export interface NormalizedBackground {
  background:    Background
  transferList?: Transferable[]
  cleanup?:      () => void
}

// ---- Parser ----

export async function normalizeBackground(input: BackgroundInput): Promise<NormalizedBackground> {
  if (typeof input === 'string') {
    if (input === 'none') return { background: { kind: 'none' } }
    if (input === 'transparent') return { background: { kind: 'transparent' } }
    if (input === 'matte') return { background: { kind: 'matte' } }
    if (input === 'blur') return { background: { kind: 'blur', sigma: SIGMA_MEDIUM } }
    return { background: { kind: 'image', bitmap: await loadImageFromUrl(input) } }
  }

  if (input instanceof ImageBitmap) {
    return { background: { kind: 'image', bitmap: input } }
  }
  if (typeof HTMLImageElement !== 'undefined' && input instanceof HTMLImageElement) {
    return { background: { kind: 'image', bitmap: await createImageBitmap(input) } }
  }
  if (typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement) {
    return await parseVideo(input)
  }

  // Cast to the object-form union: by this point we've ruled out strings
  // and DOM elements, but TS can't narrow because the typeof guards above
  // are conservative (HTMLImageElement etc. have `.blur()` methods that
  // confuse `'blur' in input`).
  const obj = input as { color?: ColorInput; blur?: BlurInput; image?: ImageInput; video?: VideoInput }
  if (obj.color !== undefined) return { background: { kind: 'color', rgb: parseColor(obj.color) } }
  if (obj.blur  !== undefined) return { background: { kind: 'blur',  sigma: parseBlur(obj.blur) } }
  if (obj.image !== undefined) return { background: { kind: 'image', bitmap: await parseImage(obj.image) } }
  if (obj.video !== undefined) return await parseVideo(obj.video)

  throw new Error(`background: unrecognized input shape — ${describe(input)}`)
}

async function parseVideo(input: VideoInput): Promise<NormalizedBackground> {
  // Coerce the public VideoInput shape into setupVideoSource's input
  // (HTMLVideoElement | string | Blob). The { data, type } object form
  // wraps to a Blob first; everything else is passed through.
  let src: VideoSourceInput
  if (typeof input === 'string') {
    src = input
  } else if (typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement) {
    src = input
  } else if (input && typeof input === 'object' && 'data' in input && 'type' in input) {
    src = input.data instanceof Blob
      ? input.data
      : new Blob([input.data as BlobPart], { type: input.type })
  } else {
    throw new Error(`background.video: unrecognized shape — ${describe(input)}`)
  }
  const setup = await setupVideoSource(src)
  return {
    background:   { kind: 'video', port: setup.port },
    transferList: setup.transferList,
    cleanup:      setup.cleanup,
  }
}

function parseColor(c: ColorInput): [number, number, number] {
  if (typeof c === 'string') {
    const hex = c.trim().replace(/^#/, '')
    if (/^[0-9a-fA-F]{6}$/.test(hex)) {
      return [
        parseInt(hex.slice(0, 2), 16) / 255,
        parseInt(hex.slice(2, 4), 16) / 255,
        parseInt(hex.slice(4, 6), 16) / 255,
      ]
    }
    if (/^[0-9a-fA-F]{3}$/.test(hex)) {
      return [
        parseInt(hex[0] + hex[0], 16) / 255,
        parseInt(hex[1] + hex[1], 16) / 255,
        parseInt(hex[2] + hex[2], 16) / 255,
      ]
    }
    throw new Error(`background.color: hex must be #rgb or #rrggbb, got ${describe(c)}`)
  }
  if (!Array.isArray(c) || c.length !== 3 || !c.every(n => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1))
    throw new Error(`background.color: expected hex string or [r, g, b] floats in [0, 1], got ${describe(c)}`)
  return [c[0], c[1], c[2]]
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
      return SIGMA_HIGH * t                 // 0 = no blur, 1 = max (SIGMA_HIGH)
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
