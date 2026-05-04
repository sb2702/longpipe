import type { Backend } from '~/model/backend'
import type { ModelWeights } from '~/model/weights'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { EfficientNetLiteMattingLarge } from '~/model/networks/efficientnetlite_matting_large'
import { BilinearUpscaler } from '~/model/effects/upscale_bilinear'
import { CompositorSolid } from '~/model/effects/compositor_solid'
import { CompositorImage } from '~/model/effects/compositor_image'
import { CompositorBlur }  from '~/model/effects/compositor_blur'
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

function drawCovered(img: ImageBitmap, w: number, h: number): Uint8ClampedArray {
  const c = new OffscreenCanvas(w, h)
  const ctx = c.getContext('2d')!
  const scale = Math.max(w / img.width, h / img.height)
  const dw = img.width * scale, dh = img.height * scale
  const dx = (w - dw) / 2, dy = (h - dh) / 2
  ctx.drawImage(img, dx, dy, dw, dh)
  return ctx.getImageData(0, 0, w, h).data
}

function bytesToFloat(rgba: Uint8ClampedArray): Float32Array {
  const out = new Float32Array(rgba.length)
  for (let i = 0; i < rgba.length; i++) out[i] = rgba[i] / 255
  return out
}

function drawBytesToCanvas(rgba: Uint8ClampedArray, canvas: HTMLCanvasElement, w: number, h: number) {
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  const id = ctx.createImageData(w, h)
  id.data.set(rgba)
  ctx.putImageData(id, 0, 0)
}

// ── Backend creation ──────────────────────────────────────────────────────

async function createBackend(name: string, canvas: HTMLCanvasElement): Promise<Backend> {
  if (name === 'webgpu') {
    if (!('gpu' in navigator)) throw new Error('WebGPU not available')
    return WebGPUBackend.create({ canvas })
  }
  return WebGLBackend.create({ canvas })
}

function parseHexColor(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255
  const g = parseInt(hex.slice(3, 5), 16) / 255
  const b = parseInt(hex.slice(5, 7), 16) / 255
  return [r, g, b]
}

// ── Pipeline ──────────────────────────────────────────────────────────────

async function run() {
  const backendName = (document.getElementById('backend') as HTMLSelectElement).value
  const bgMode      = (document.getElementById('bgMode')  as HTMLSelectElement).value
  const bgHex       = (document.getElementById('bgColor') as HTMLInputElement).value
  const sigma       = parseFloat((document.getElementById('bgSigma') as HTMLInputElement).value)
  const bgColor     = parseHexColor(bgHex)

  const inputCanvas  = document.getElementById('inputCanvas')  as HTMLCanvasElement

  const oldOutput = document.getElementById('outputCanvas') as HTMLCanvasElement
  const outputCanvas = document.createElement('canvas')
  outputCanvas.id = 'outputCanvas'
  oldOutput.replaceWith(outputCanvas)

  status(`loading test image and weights (${backendName}, bg=${bgMode})…`)
  const fetches: Array<Promise<unknown>> = [
    loadImage('/test_img.jpg'),
    fetch('/model_large.bin').then(r => {
      if (!r.ok) throw new Error(`failed to fetch /model_large.bin: ${r.status}`)
      return r.arrayBuffer()
    }),
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

  status(`creating ${backendName} backend (${dispW}×${dispH} canvas)…`)
  const backend = await createBackend(backendName, outputCanvas)

  status('preparing tensors…')
  const dispBytes = drawCovered(img, dispW, dispH)
  const netBytes  = drawCovered(img, NET_W, NET_H)
  drawBytesToCanvas(netBytes, inputCanvas, NET_W, NET_H)

  const dispTensor  = backend.tensor(dispH, dispW, 4, bytesToFloat(dispBytes))
  const inputTensor = backend.tensor(NET_H, NET_W, 4, bytesToFloat(netBytes))

  status('parsing weights…')
  const weights = loadWeightsFromBinary(weightsBuf) as ModelWeights

  status('building network…')
  const model = new EfficientNetLiteMattingLarge(backend, inputTensor, weights)

  status('running network…')
  const t0 = performance.now()
  model.run()
  const tNet = performance.now() - t0

  status('upscaling alpha…')
  const upscaler = new BilinearUpscaler(backend, model.output, dispH, dispW)
  upscaler.run()

  status(`compositing (${bgMode})…`)
  const tComp0 = performance.now()
  if (bgMode === 'solid') {
    new CompositorSolid(backend, dispTensor, upscaler.output, bgColor).run()
  } else if (bgMode === 'image') {
    if (!bgImg) throw new Error('bg image failed to load')
    const bgBytes  = drawCovered(bgImg, dispW, dispH)
    const bgTensor = backend.tensor(dispH, dispW, 4, bytesToFloat(bgBytes))
    new CompositorImage(backend, dispTensor, upscaler.output, bgTensor).run()
  } else {
    new CompositorBlur(backend, dispTensor, upscaler.output, sigma).run()
  }
  const tComp = performance.now() - tComp0

  status(`done. network=${tNet.toFixed(1)}ms · compose=${tComp.toFixed(1)}ms · canvas=${dispW}×${dispH} · bg=${bgMode}`)
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
