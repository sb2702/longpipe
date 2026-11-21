// Downloads the media + weights the vanilla-WebRTC example and the e2e smoke
// tests load, pinned by URL + sha256, into examples/vanilla-webrtc/public/assets/
// (gitignored). Same approach as benchmarks/performance/tools/fetch_assets.mjs:
// nothing third-party is committed, and a run can't silently pick up a newer
// clip or model. Safe to re-run (skips files whose hash already matches).
//
//   node tests/e2e/fetch_assets.mjs          (also: npm run e2e:assets)
//
// The weights are served locally so the tests don't depend on (or show up in
// the logs of) cdn.longpipe.dev, and so a run is reproducible offline.
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

const OUT = new URL('../../examples/vanilla-webrtc/public/assets/', import.meta.url).pathname

// The test clip: a person on a video call (640x360 H.264, 192 frames @ 24 fps).
// Recorded by Sam Bhattacharyya (Longpipe's author); the same clip and hash as
// the performance benchmark.
const CLIP_BASE = 'https://raw.githubusercontent.com/sb2702/webrtc-hacks-webgpu/2c6528cf3c87f726c946ca24c1aa096ea5ef2db1/segmentation/web/public/media'
const LP_BASE = 'https://cdn.longpipe.dev/models/v/0.0.5'   // DEFAULT_WEIGHTS_BASE_URL (src/weights_url.ts)

export const ASSETS = {
  'media/woman_on_video_call_360p.mp4': [`${CLIP_BASE}/woman_on_video_call_360p.mp4`, 'bb0b4b03a287ebc1b140e64ca1fa6748a3bd3647868a3e201d13654c871b20ee'],
  // The e2e tests pin preset 'xs' (fast on software GPUs). f16 pack for
  // shader-f16 adapters and WebGL2; f32 .bin for WebGPU adapters without f16.
  'models/model_xs.f16.bin': [`${LP_BASE}/model_xs.f16.bin`, 'f1786bf9ffb03916fd265c253f4928ec4a7576855162aacc856aaf90908899e9'],
  'models/model_xs.bin':     [`${LP_BASE}/model_xs.bin`,     '501cd8fc26a260552c62d22cde4d5ab7b32087921f42ba2539a9b5300d3f8c9d'],
  // Audio denoise: the e2e tests pin model 'rnnoise' (weights embedded in the wasm).
  'models/rnnoise.wasm':     [`${LP_BASE}/rnnoise.wasm`,     'c2b6449c5f3d3136c9819dd23211907233b6cfd2b2195082463ba26c879e1910'],
  // DFN (int8 pack, the smaller one): the worklet unit test (tests/audio/worklet_output.test.ts).
  'models/dfn.wasm':              [`${LP_BASE}/dfn.wasm`,              'bb06861ecb54e94c178c80225c99c4b13aa6113943aa2f5c782b341d5ed27998'],
  'models/dfn_weights_int8.pack': [`${LP_BASE}/dfn_weights_int8.pack`, '57ddb56075bd48ce28876aa3295fbd13f6ee0c5dc50100df2a1af526b1f5f71e'],
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')

for (const [path, [url, digest]] of Object.entries(ASSETS)) {
  const out = OUT + path
  if (existsSync(out) && sha256(readFileSync(out)) === digest) continue
  process.stdout.write(`fetching ${url}\n`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const buf = Buffer.from(await res.arrayBuffer())
  const got = sha256(buf)
  if (got !== digest) throw new Error(`${url}: sha256 ${got}, expected ${digest}`)
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, buf)
}
console.log(`e2e assets ready in ${OUT}`)

// VP9 WebM copy of the clip for browsers without H.264 (Firefox on Linux without system
// ffmpeg libraries). Derived locally from the hash-checked MP4; skipped if ffmpeg is missing.
{
  const { execFileSync } = await import('node:child_process')
  const mp4 = OUT + 'media/woman_on_video_call_360p.mp4'
  const webm = OUT + 'media/woman_on_video_call_360p.webm'
  if (!existsSync(webm)) {
    try {
      execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', mp4, '-an', '-c:v', 'libvpx-vp9', '-b:v', '1M', '-deadline', 'good', '-cpu-used', '4', webm], { stdio: 'inherit' })
      process.stdout.write('made media/woman_on_video_call_360p.webm (VP9)\n')
    } catch {
      process.stdout.write('ffmpeg not found: skipping the VP9 WebM copy (only needed for Firefox without H.264)\n')
    }
  }
}
