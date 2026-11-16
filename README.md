# Longpipe 🐉

Fast, high-quality video effects in the browser — virtual backgrounds, face touch-up, auto-reframe, and mic noise removal in one open-source SDK.

<img width="960" height="540" alt="video-effects-sdk" src="https://github.com/user-attachments/assets/5dbe1ccb-09dd-4f22-b4be-e3122767dbcc" />

**Try the live demo → [longpipe.dev/demo](https://longpipe.dev/demo)**

> **Warning** — This project is very new and still under active development. Expect API changes between versions and bugs.

## Quick start

```
npm install longpipe
```

```ts
import { EffectsPipeline } from 'longpipe'

const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })

const pipeline = new EffectsPipeline(stream, {
  background: 'blur',   // or an image, video, or solid color
  touchup: true,        // skin smoothing
  reframe: true,        // auto-frame the subject
  audio: 'denoise',     // mic noise removal
})

videoEl.srcObject = pipeline.stream   // available immediately
await pipeline.ready                  // optional — resolves once the effect is live
```

`pipeline.stream` is wired synchronously and emits the unprocessed input until the model is ready (~1–3 s on cold start), so users see live video the whole time. Model weights stream from `cdn.longpipe.dev` by default — no extra setup, and [self-hostable](https://longpipe.dev/docs/self-hosting).

## Why Longpipe

- **One model, many effects.** Backgrounds, touch-up, and auto-reframe all run off a single shared encoder pass — enabling another effect doesn't cost another inference. Audio denoise runs in a parallel `AudioWorklet`, off the GPU entirely. ([Architecture](https://longpipe.dev/docs/architecture))
- **Lower matting error than MediaPipe's selfie segmenter at every preset, and faster than MediaPipe Tasks on WebGPU at xs–medium.** Custom-trained models running as pure WebGPU/WebGL shaders in a zero-copy, fully-GPU pipeline — no general-purpose runtime, no CPU↔GPU round trips. Numbers, caveats and where MediaPipe wins (its model on LiteRT.js is faster): [benchmarks](#benchmarks-vs-mediapipe).
- **Works everywhere.** Chromium, Firefox, and Safari (desktop + iOS). WebGPU when available, WebGL2 fallback, per-browser frame transport handled internally.
- **Adapts to the device.** Five model presets (`xs`–`xl`); autotune benchmarks the actual device at init and an adaptive controller swaps presets at runtime to hold 30 fps — on 10-year-old netbooks and the latest MacBook Pro alike.
- **Production pedigree.** Built by the founder of [Vectorly](https://www.crunchbase.com/organization/vectorly), a commercial effects SDK acquired in 2021 — a ground-up redesign for the WebGPU era.

### Benchmarks vs MediaPipe

Reproducible benchmarks against Google's MediaPipe selfie segmenter, with code, raw results and methodology in [`benchmarks/`](benchmarks/). Read [the methodology](benchmarks/METHODOLOGY.md) for the caveats before quoting a number.

**Quality**: P3M-10k validation sets (1,000 portrait photos with labelled alpha mattes), scored at full resolution with the P3M paper's metric code. MAD = mean absolute alpha error (lower is better); IoU of the mask at 0.5 (higher is better).

| Model | Mask | P3M-500-P MAD | P3M-500-NP MAD | P3M-500-P IoU | P3M-500-NP IoU |
|---|---|---:|---:|---:|---:|
| Longpipe xs | 384×240 | 0.0210 | 0.0233 | 0.923 | 0.926 |
| Longpipe small | 384×224 | 0.0170 | 0.0155 | 0.937 | 0.954 |
| Longpipe medium | 512×320 | 0.0106 | 0.0102 | 0.966 | 0.972 |
| Longpipe large | 640×400 | 0.0111 | 0.0100 | 0.963 | 0.972 |
| Longpipe xl | 1280×768 | 0.0071 | 0.0080 | 0.977 | 0.979 |
| MediaPipe selfie (square) | 256×256 | 0.0304 | 0.0249 | 0.923 | 0.940 |
| MediaPipe selfie (landscape) | 256×144 | 0.0322 | 0.0276 | 0.917 | 0.934 |

Every preset has lower alpha error than MediaPipe, but MediaPipe's square model beats **xs** on IoU on P3M-500-NP and ties it on P3M-500-P. Longpipe trains on P3M-10k's training split and MediaPipe doesn't, so P3M-500-NP (un-blurred faces, not used for model selection) is the fairer of the two sets. MediaPipe is a segmentation model scored on a matting task; part of the gap is mask resolution.

In the speed table below, MediaPipe's model on other engines (LiteRT.js, Longpipe's) has the MediaPipe scores above: same weights. Longpipe's port reproduces the TFLite output to a max abs error of 1.4e-5 in f32 (mean ~5e-5 in its default f16).

**Speed**: background replacement on an Apple M4 MacBook Pro, Chrome 157: WebCodecs decode → effect → H.264 encode of a 192-frame webcam clip, frames per second, median of 5 runs, two sessions (s1 / s2). Longpipe "as shipped" is the SDK's own per-frame renderer: the full network on key frames (every 3rd frame for xs, every 2nd for small/medium, every frame for large/xl), a small learned optical-flow network on the frames between, and a temporal stabiliser on every frame.

| Longpipe model, as shipped | xs | small | medium | large | xl |
|---|---:|---:|---:|---:|---:|
| WebGPU, 360p | 308 / 288 | 291 / 298 | 241 / 237 | 150 / 157 | 91 / 88 |
| WebGPU, 720p | 159 / 161 | 155 / 162 | 140 / 146 | 104 / 108 | 69 / 69 |
| WebGL2, 360p | 205 / 192 | 195 / 189 | 154 / 159 | 102 / 101 | 63 / 60 |
| WebGL2, 720p | 122 / 129 | 121 / 125 | 104 / 105 | 76 / 78 | 49 / 52 |

| MediaPipe selfie model: square s1 / s2 · landscape s1 / s2 | 360p | 720p |
|---|---:|---:|
| MediaPipe Tasks, GPU delegate | 118 / 114 · 123 / 120 | 93 / 94 · 93 / 97 |
| MediaPipe Tasks, CPU delegate | 117 / 114 · 150 / 153 | 69 / 70 · 81 / 82 |
| LiteRT.js, WebGPU | 380 / 381 · 414 / 411 | 201 / 196 · 196 / 204 |
| Longpipe engine (`model: 'mediapipe-selfie'`), WebGPU | 206 / 192 · 209 / 205 | 132 / 130 · 130 / 132 |
| Longpipe engine, WebGL2 | 125 / 123 · 122 / 123 | 90 / 90 · 88 / 92 |

On this machine, Longpipe xs–medium on WebGPU are faster than MediaPipe Tasks in every configuration, large is level with its best (360p) or faster (720p), and xl is slower than its best; on WebGL2, xs–small are faster, medium level, and large and xl slower than its best. **MediaPipe's model running on LiteRT.js is faster than every Longpipe preset.** The flow schedule is 0–14% faster than running the network on every frame for xs–medium, and 18–27% slower for large/xl. Full tables, the network-every-frame rows and what each pipeline runs: [`benchmarks/`](benchmarks/README.md#performance-apple-m4-macbook-pro-chrome-157-v2). It's one machine, in use while measured; run [the benchmark](benchmarks/README.md) on yours.

## Models

Longpipe runs one of two segmentation models through the same GPU pipeline (WebGPU / WebGL2 backends, zero-copy frame input, compositor, worker and transports):

| `model` | What it is | Background effects | Touch-up, auto-reframe | Flow stabilization | Presets / autotune |
|---|---|---|---|---|---|
| `'longpipe'` (default) | Longpipe's own hydranet, five tiers `xs`–`xl` | ✓ | ✓ | ✓ | ✓ |
| `'mediapipe-selfie'` | Google's [MediaPipe Selfie Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#selfie-model), square 256×256 or landscape 256×144 (picked from the output aspect; force with `'mediapipe-selfie-square'` / `'mediapipe-selfie-landscape'`) | ✓ | ✗ (throws) | ✗ | ✗ (one fixed model) |

```ts
const pipeline = new EffectsPipeline(stream, { model: 'mediapipe-selfie', background: 'blur' })
```

Use `'mediapipe-selfie'` if you specifically want MediaPipe's model, e.g. to match an existing MediaPipe deployment while getting Longpipe's pipeline. Longpipe's own models drive every effect from one encoder pass, and are more accurate (lower error on P3M-10k, even at `xs`) and faster on the same engine (`xs` network: 1.6 ms/frame vs 2.6–2.7 ms for MediaPipe's model on the WebGPU backend, Apple M4; [benchmarks](benchmarks/README.md)). The MediaPipe weights are Apache-2.0 (© Google) and are converted by [`scripts/convert_mediapipe_selfie.py`](scripts/convert_mediapipe_selfie.py); see [`third_party/mediapipe/NOTICE.md`](third_party/mediapipe/NOTICE.md). They are served from `cdn.longpipe.dev/models/mediapipe/selfie/v1/`, or from your `weightsBaseUrl` if you self-host.

## Documentation

Full docs live at **[longpipe.dev/docs](https://longpipe.dev/docs)**:

- [Getting started](https://longpipe.dev/docs/getting-started)
- [API reference](https://longpipe.dev/docs/api) — all `EffectsPipeline` options and methods
- [Architecture](https://longpipe.dev/docs/architecture) — the hydranet: one encoder, many heads
- [Backgrounds](https://longpipe.dev/docs/backgrounds) · [Touch-up](https://longpipe.dev/docs/touchup) · [Auto-reframe](https://longpipe.dev/docs/reframe) · [Audio denoise](https://longpipe.dev/docs/audio)
- [Presets & autotune](https://longpipe.dev/docs/presets)
- [Model layer](https://longpipe.dev/docs/model-layer) — drive the model directly, without the pipeline
- [Self-hosting weights](https://longpipe.dev/docs/self-hosting)
- [ONNX & other runtimes](https://longpipe.dev/docs/onnx)

Machine-readable for LLMs and agents: [`llms.txt`](https://longpipe.dev/llms.txt) · [`llms-full.txt`](https://longpipe.dev/llms-full.txt)

## Models (ONNX)

The trained models are also published as official ONNX exports (all five tiers, matting, face keypoints,
optical flow and the 478-point face mesh) for Python, native apps, servers and `onnxruntime-web`:

- **Hugging Face:** [huggingface.co/Longpipe/longpipe](https://huggingface.co/Longpipe/longpipe)
- **CDN:** `https://cdn.longpipe.dev/models/v/0.0.5/onnx/` (CORS-enabled)

Inputs, outputs and examples: [longpipe.dev/docs/onnx](https://longpipe.dev/docs/onnx).

## Roadmap

- [x] Background segmentation / virtual backgrounds
- [x] Background noise removal (audio, separate pipeline)
- [x] Face landmarks + touch-up
- [x] Multi-face support
- [x] Auto-reframe
- [ ] AR effects
- [ ] Lighting correction

## License

SDK source code **and** Longpipe's pre-trained model weights are MIT-licensed — see [LICENSE](LICENSE) and [WEIGHTS_LICENSE](WEIGHTS_LICENSE). You're free to self-host the weights inside other open-source or commercial projects under MIT-compatible terms. The optional MediaPipe selfie weights are Google's, under Apache-2.0 ([NOTICE](third_party/mediapipe/NOTICE.md)).

The weights were trained on permissively-licensed public datasets (P3M-10k, AISegment, COCO, OpenImages), Z-Image synthetics, and a custom webcam dataset collected with participants' explicit, informed consent. The full dataset list and provenance details: [longpipe.dev/docs/licensing](https://longpipe.dev/docs/licensing).
