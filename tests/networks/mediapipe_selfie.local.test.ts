import { describe, it, expect } from 'vitest'
import { MediaPipeSelfie, MEDIAPIPE_SELFIE_CONFIG, type MediaPipeSelfieWeights } from '~/model/networks/mediapipe_selfie'
import { loadWeightsFromBinary } from '~/utils/loadWeights'
import { BACKENDS_DTYPE } from '../helpers/backends'

// MediaPipe selfie segmenter, end to end on the GPU backends vs the TFLite
// interpreter. Artifacts are LOCAL (not committed); generate them with
//   python3 scripts/convert_mediapipe_selfie.py --image <a frame with a person>
// which stages the weight packs in dist-weights/ and writes the golden
// input/mask pairs to tests/fixtures/local/. Self-skips when they're absent.
const refUrls = (import.meta as any).glob('../fixtures/local/mediapipe_selfie_*.ref.json',
  { query: '?url', import: 'default', eager: true }) as Record<string, string>
const binUrls = (import.meta as any).glob('../../dist-weights/mediapipe/selfie/v1/*.bin',
  { query: '?url', import: 'default', eager: true }) as Record<string, string>

const cases = Object.entries(refUrls).map(([path, refUrl]) => {
  const variant = path.match(/mediapipe_selfie_(\w+)\.ref\.json$/)![1] as keyof typeof MEDIAPIPE_SELFIE_CONFIG
  const file = MEDIAPIPE_SELFIE_CONFIG[variant]?.file
  const binUrl = Object.entries(binUrls).find(([p]) => p.endsWith('/' + file))?.[1]
  return { variant, refUrl, binUrl: binUrl! }
}).filter(c => c.binUrl)

// f32 must match TFLite tightly. f16 storage/compute is reported, with a loose
// bound (the pipeline runs this model in f32).
const LIMITS = { f32: { max: 1e-3, mean: 1e-5 }, f16: { max: 0.25, mean: 5e-3 } }

describe.each(BACKENDS_DTYPE)('MediaPipe selfie vs TFLite ($name)', ({ name, dtype, create }) => {
  if (cases.length === 0) {
    it.skip('no local fixtures — run scripts/convert_mediapipe_selfie.py', () => {})
    return
  }
  it.each(cases)('$variant', async ({ variant, refUrl, binUrl }) => {
    let backend
    try { backend = await create() } catch (e) { if (dtype === 'f16') return; throw e }   // no shader-f16
    const ref = await (await fetch(refUrl)).json() as { h: number; w: number; input: number[]; expected: number[] }
    const w = loadWeightsFromBinary(await (await fetch(binUrl)).arrayBuffer()) as unknown as MediaPipeSelfieWeights

    const { h, w: wd } = ref
    const rgba = new Float32Array(h * wd * 4)
    for (let p = 0; p < h * wd; p++) for (let c = 0; c < 3; c++) rgba[p * 4 + c] = ref.input[p * 3 + c]
    const net = new MediaPipeSelfie(backend, backend.tensor(h, wd, 4, rgba), w)
    net.run()
    const out = await backend.readback(net.output)
    backend.destroy()

    let max = 0, sum = 0
    for (let p = 0; p < h * wd; p++) {
      const d = Math.abs(out[p * 4] - ref.expected[p])
      max = Math.max(max, d); sum += d
    }
    const mean = sum / (h * wd)
    console.log(`[mediapipe parity] ${name} ${variant} ${wd}x${h}: max ${max.toExponential(2)} mean ${mean.toExponential(2)} (${net.opCount} ops)`)
    expect(max).toBeLessThan(LIMITS[dtype].max)
    expect(mean).toBeLessThan(LIMITS[dtype].mean)
  })
})
