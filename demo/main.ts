import type { Backend, Dtype } from '~/model/backend'
import type { ModelWeights } from '~/model/weights'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { EfficientNetLiteMattingLarge } from '~/model/networks/efficientnetlite_matting_large'
import { RenderOp, type BackgroundConfig } from '~/model/render_op'
import { loadWeightsFromBinary } from '~/utils/loadWeights'

// Network input dimensions for the "large" preset (16:9 landscape).
const NET_H = 144
const NET_W = 256

const status = (s: string) => {
  const el = document.getElementById('status')!
  el.textContent = s
  console.log('[demo]', s)
}

// ── Image loading ─────────────────────────────────────────────────────────

async function loadImage(url: string): Promise<ImageBitmap> {
  const blob = await fetch(url).then(r => {
    if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`)
    return r.blob()
  })
  return createImageBitmap(blob)
}

// Crop the bitmap to a target aspect ratio at native resolution (cover-style:
// preserve the maximum centred area). The Input op only does stretch, so the
// caller does the aspect math here.
async function cropToAspect(img: ImageBitmap, aspect: number): Promise<ImageBitmap> {
  const srcAspect = img.width / img.height
  let sx: number, sy: number, sw: number, sh: number
  if (srcAspect > aspect) {
    sh = img.height
    sw = Math.round(sh * aspect)
    sx = Math.round((img.width - sw) / 2)
    sy = 0
  } else {
    sw = img.width
    sh = Math.round(sw / aspect)
    sx = 0
    sy = Math.round((img.height - sh) / 2)
  }
  return createImageBitmap(img, sx, sy, sw, sh)
}

// ── Backend creation ──────────────────────────────────────────────────────

async function createBackend(name: string, dtype: Dtype, canvas: HTMLCanvasElement): Promise<Backend> {
  if (name === 'webgpu') {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    if (dtype === 'f16' && !(await WebGPUBackend.hasF16Support())) {
      throw new Error("WebGPU adapter lacks `shader-f16` feature — try Chrome with --enable-unsafe-webgpu, or pick a different precision/backend")
    }
    return WebGPUBackend.create({ canvas, dtype })
  }
  return WebGLBackend.create({ canvas, dtype })
}

// Try the dtype-matched .f16.bin first; fall back to the fp32 .bin (backend
// converts at upload-time). Lets the demo work whether or not the user has
// regenerated weights with the latest Python serializer.
//
// Vite's dev server returns index.html for missing files (SPA fallback), so
// `r.ok` alone isn't enough — also check the content-type to detect the HTML
// fallback before treating the response as a binary blob.
async function fetchBin(url: string): Promise<ArrayBuffer | null> {
  const r = await fetch(url)
  if (!r.ok) return null
  const ct = r.headers.get('content-type') ?? ''
  if (ct.startsWith('text/html')) return null
  return r.arrayBuffer()
}

async function fetchWeights(dtype: Dtype): Promise<ArrayBuffer> {
  if (dtype === 'f16') {
    const f16 = await fetchBin('/model_large.f16.bin')
    if (f16) return f16
    console.warn('[demo] no model_large.f16.bin — falling back to fp32 .bin (backend will convert)')
  }
  const f32 = await fetchBin('/model_large.bin')
  if (!f32) throw new Error('failed to fetch /model_large.bin')
  return f32
}

function parseHexColor(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255
  const g = parseInt(hex.slice(3, 5), 16) / 255
  const b = parseInt(hex.slice(5, 7), 16) / 255
  return [r, g, b]
}

// ── Pipeline ──────────────────────────────────────────────────────────────

async function run() {
  const backendName  = (document.getElementById('backend')  as HTMLSelectElement).value
  const dtype        = (document.getElementById('dtype')    as HTMLSelectElement).value as Dtype
  const upscalerMode = (document.getElementById('upscaler') as HTMLSelectElement).value
  const bgMode       = (document.getElementById('bgMode')   as HTMLSelectElement).value
  const bgHex       = (document.getElementById('bgColor') as HTMLInputElement).value
  const sigma       = parseFloat((document.getElementById('bgSigma') as HTMLInputElement).value)
  const bgColor     = parseHexColor(bgHex)

  const inputCanvas  = document.getElementById('inputCanvas')  as HTMLCanvasElement

  const oldOutput = document.getElementById('outputCanvas') as HTMLCanvasElement
  const outputCanvas = document.createElement('canvas')
  outputCanvas.id = 'outputCanvas'
  oldOutput.replaceWith(outputCanvas)

  status(`loading test image and weights (${backendName}/${dtype}, bg=${bgMode})…`)
  const fetches: Array<Promise<unknown>> = [
    loadImage('/test_img.jpg'),
    fetchWeights(dtype),
  ]
  if (bgMode === 'image') fetches.push(loadImage('/demo.jpg'))
  const [img, weightsBuf, bgImg] = await Promise.all(fetches) as [ImageBitmap, ArrayBuffer, ImageBitmap?]

  // Display resolution: largest 16:9 box that fits the input image, capped.
  const targetAspect = NET_W / NET_H
  const srcAspect = img.width / img.height
  let dispW: number, dispH: number
  if (srcAspect > targetAspect) {
    dispH = img.height
    dispW = Math.round(dispH * targetAspect)
  } else {
    dispW = img.width
    dispH = Math.round(dispW / targetAspect)
  }
  const cap = 1024
  if (dispW > cap) {
    dispH = Math.round(dispH * cap / dispW)
    dispW = cap
  }
  outputCanvas.width  = dispW
  outputCanvas.height = dispH

  status(`creating ${backendName}/${dtype} backend (${dispW}×${dispH} canvas)…`)
  const backend = await createBackend(backendName, dtype, outputCanvas)

  status('cropping source…')
  // Crop once to network aspect; the Input ops inside RenderOp bilinear-
  // resample to display + network resolutions. Same ImageBitmap drives both;
  // in the pipeline layer this would be the same VideoFrame each tick.
  const cropped = await cropToAspect(img, targetAspect)

  // Visualize the network-input source on the small input canvas.
  inputCanvas.width  = NET_W
  inputCanvas.height = NET_H
  inputCanvas.getContext('2d')!.drawImage(cropped, 0, 0, NET_W, NET_H)

  status('parsing weights…')
  const weights = loadWeightsFromBinary(weightsBuf) as ModelWeights

  status('building network…')
  // Network owns its input tensor (constructed at network resolution); we
  // hand the same Input op to RenderOp so its setSource() can fan out.
  const networkInput = backend.ops.Input(NET_H, NET_W)
  const model = new EfficientNetLiteMattingLarge(backend, networkInput.output, weights)

  // Resolve background config — for image mode, ingest demo.jpg via a one-
  // shot Input op (static, no per-frame fanout needed).
  let bgConfig: BackgroundConfig
  if (bgMode === 'solid') {
    bgConfig = { mode: 'solid', color: bgColor }
  } else if (bgMode === 'image') {
    if (!bgImg) throw new Error('bg image failed to load')
    const bgCropped = await cropToAspect(bgImg, targetAspect)
    const bgInput = backend.ops.Input(dispH, dispW)
    bgInput.setSource(bgCropped)
    bgInput.run()
    bgConfig = { mode: 'image', image: bgInput.output }
  } else {
    bgConfig = { mode: 'blur', sigma }
  }

  const renderOp = new RenderOp(backend, model, networkInput, {
    upscaler:   upscalerMode === 'bicubic' ? 'bicubic' : 'bilinear',
    background: bgConfig,
  })

  status('rendering…')
  const t0 = performance.now()
  renderOp.setSource(cropped)
  renderOp.run()
  const tTotal = performance.now() - t0

  status(`done. render=${tTotal.toFixed(1)}ms · ${backendName}/${dtype} · canvas=${dispW}×${dispH} · upscale=${upscalerMode} · bg=${bgMode}`)
}

// ── UI wiring ─────────────────────────────────────────────────────────────

const bgModeSelect  = document.getElementById('bgMode')        as HTMLSelectElement
const bgColorLabel  = document.getElementById('bgColorLabel')  as HTMLLabelElement
const bgSigmaLabel  = document.getElementById('bgSigmaLabel')  as HTMLLabelElement
const bgSigmaInput  = document.getElementById('bgSigma')       as HTMLInputElement
const bgSigmaVal    = document.getElementById('bgSigmaVal')    as HTMLSpanElement

function syncBgControls() {
  const mode = bgModeSelect.value
  bgColorLabel.style.display = mode === 'solid' ? '' : 'none'
  bgSigmaLabel.style.display = mode === 'blur'  ? '' : 'none'
}
bgModeSelect.addEventListener('change', syncBgControls)
bgSigmaInput.addEventListener('input', () => { bgSigmaVal.textContent = bgSigmaInput.value })
syncBgControls()

document.getElementById('run')!.addEventListener('click', () => {
  run().catch(err => {
    console.error(err)
    status(`error: ${err.message ?? err}`)
  })
})

run().catch(err => {
  console.error(err)
  status(`error: ${err.message ?? err}`)
})
