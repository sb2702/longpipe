// Downloads everything the benchmark page loads, pinned by URL + sha256, into
// web/public/assets/, and copies the MediaPipe / LiteRT / web-demuxer WASM out of
// node_modules into web/public/vendor/. Runs on `npm install`; safe to re-run.
//
// Nothing is fetched at benchmark time, so a run can't silently pick up a newer
// model or clip.
import { createHash } from 'node:crypto'
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

const root = new URL('../', import.meta.url).pathname

// The test clip: a person on a video call, 1280x720 H.264, 192 frames @ 24 fps,
// and a 640x360 re-encode. Recorded by Sam Bhattacharyya (Longpipe's author) and
// published with the webrtcHacks WebGPU article benchmark this harness is based on.
const CLIP_BASE = 'https://raw.githubusercontent.com/sb2702/webrtc-hacks-webgpu/2c6528cf3c87f726c946ca24c1aa096ea5ef2db1/segmentation/web/public/media'
const MP_BASE = 'https://storage.googleapis.com/mediapipe-models/image_segmenter'
const LP_BASE = 'https://cdn.longpipe.dev/models/v/0.0.5'   // the SDK's default weights URL (src/weights_url.ts)
// MediaPipe selfie weights converted for Longpipe's engine (DEFAULT_MEDIAPIPE_WEIGHTS_BASE_URL, src/weights_url.ts)
const LP_MP_BASE = 'https://cdn.longpipe.dev/models/mediapipe/selfie/v1'

export const ASSETS = {
  'media/woman_on_video_call_720p.mp4': [`${CLIP_BASE}/woman_on_video_call.mp4`, '28375ef0dbed9d8d4d5ae0f9866f6b39d9a003580979dab91546c8dab92b8d5c'],
  'media/woman_on_video_call_360p.mp4': [`${CLIP_BASE}/woman_on_video_call_360p.mp4`, 'bb0b4b03a287ebc1b140e64ca1fa6748a3bd3647868a3e201d13654c871b20ee'],
  'models/selfie_segmenter.tflite': [`${MP_BASE}/selfie_segmenter/float16/1/selfie_segmenter.tflite`, '191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b'],
  'models/selfie_segmenter_landscape.tflite': [`${MP_BASE}/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite`, '490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e'],
  'models/model_xs.f16.bin': [`${LP_BASE}/model_xs.f16.bin`, 'f1786bf9ffb03916fd265c253f4928ec4a7576855162aacc856aaf90908899e9'],
  'models/model_small.f16.bin': [`${LP_BASE}/model_small.f16.bin`, '8fea35e96ee8e37d1b2b6e901a2a7f8df1ce2ec14e1a1227636df6b410e2c11f'],
  'models/model_medium.f16.bin': [`${LP_BASE}/model_medium.f16.bin`, '7d8bf5d787da24e679193e70db1e7b9ca62f473ef79d8f14db1604dda8d99dd5'],
  'models/model_large.bin': [`${LP_BASE}/model_large.bin`, '5294fbbfd1eac70a6ee8d99dbe4de9a6d8c459d2c8a893c24402b35464e121f5'],
  'models/model_xl.bin': [`${LP_BASE}/model_xl.bin`, 'c1ea11f3ff2a17eb60fe53c5d37d91c410f6d992ecb245236b21c7ef80217032'],
  // EffectsPipeline fetches the .f16.bin for every preset when the backend runs f16 (its default), large/xl included.
  'models/model_large.f16.bin': [`${LP_BASE}/model_large.f16.bin`, '40ad8b9f538cf70b89b09a41f916ab441610c9a92ff4222a27071fa22fe92ae9'],
  'models/model_xl.f16.bin': [`${LP_BASE}/model_xl.f16.bin`, '4f25cb0c267d8966329ea7882afd42a64856786d5de1a276b67eb20f42043fe4'],
  'models/mediapipe_selfie_segmenter.f16.bin': [`${LP_MP_BASE}/mediapipe_selfie_segmenter.f16.bin`, 'b0f9be147827c08667fa8eb46c7cb350c8b4978e657b55214e896a9ea69a4c4d'],
  'models/mediapipe_selfie_segmenter_landscape.f16.bin': [`${LP_MP_BASE}/mediapipe_selfie_segmenter_landscape.f16.bin`, 'a74992eb5f952397d69387aab45c80f6faaec73a9e8bfae263ea28df383fc0cc'],
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')

for (const [path, [url, digest]] of Object.entries(ASSETS)) {
  const out = `${root}web/public/assets/${path}`
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

const copies = {
  'node_modules/@litertjs/core/wasm': 'web/public/vendor/litert',
  'node_modules/@mediapipe/tasks-vision/wasm': 'web/public/vendor/mediapipe',
  'node_modules/web-demuxer/dist/wasm-files/web-demuxer.wasm': 'web/public/vendor/web-demuxer.wasm',
}
mkdirSync(`${root}web/public/vendor`, { recursive: true })
for (const [from, to] of Object.entries(copies)) cpSync(root + from, root + to, { recursive: true })
console.log('assets ready')
