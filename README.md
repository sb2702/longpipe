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

EffectsPipeline.unlockAudio()   // in your click handler, before any await: lets denoise start
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

No bundler? See the single-file [minimal example](https://github.com/sb2702/longpipe/tree/main/examples/minimal) (CDN import, ~25 lines).

`pipeline.stream` is wired synchronously and emits the unprocessed input until the model is ready (~1–3 s on cold start), so users see live video the whole time. Model weights stream from `cdn.longpipe.dev` by default — no extra setup, and [self-hostable](https://longpipe.dev/docs/self-hosting).

## Why Longpipe

- **One model, many effects.** Backgrounds, touch-up, and auto-reframe all run off a single shared encoder pass — enabling another effect doesn't cost another inference. Audio denoise runs in a parallel `AudioWorklet`, off the GPU entirely. ([Architecture](https://longpipe.dev/docs/architecture))
- **More accurate than MediaPipe's selfie segmenter at every preset, and faster at `xs`–`large`.** Custom-trained models running as hand-written WebGPU/WebGL shaders in a zero-copy, fully-GPU pipeline. See the [benchmarks](#benchmarks-vs-mediapipe).
- **Works everywhere.** Chromium, Firefox, and Safari (desktop + iOS). WebGPU when available, WebGL2 fallback, per-browser frame transport handled internally.
- **Adapts to the device.** Five model presets (`xs`–`xl`); autotune benchmarks the actual device at init and an adaptive controller swaps presets at runtime to hold 30 fps — on 10-year-old netbooks and the latest MacBook Pro alike.
- **Production pedigree.** Built by the founder of [Vectorly](https://www.crunchbase.com/organization/vectorly), a commercial effects SDK acquired in 2021 — a ground-up redesign for the WebGPU era.

### Benchmarks vs MediaPipe

Longpipe implements multiple downstream features (face effects, backgrounds) with one neural network, but if you want to compare it to another library, the closest would be the Mediapipe selfie model for background segmentation, from the [@mediapipe/tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) library. 

![Frames per second, 360p, Apple M4](https://raw.githubusercontent.com/sb2702/longpipe/main/benchmarks/charts/speed-360p.svg)

![Matte error on P3M-500-P](https://raw.githubusercontent.com/sb2702/longpipe/main/benchmarks/charts/quality-p3m.svg)

Methodology, caveats (including where MediaPipe's model is faster, on LiteRT.js), the 720p, WebGL and IoU results,
and scripts to reproduce everything: **[full analysis](https://github.com/sb2702/longpipe/blob/main/benchmarks/README.md)**.

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

## Roadmap

- [x] Background segmentation / virtual backgrounds
- [x] Background noise removal (audio, separate pipeline)
- [x] Face landmarks + touch-up
- [x] Multi-face support
- [x] Auto-reframe
- [ ] AR effects
- [ ] Lighting correction

## License

SDK source code **and** Longpipe's pre-trained model weights are MIT-licensed — see [LICENSE](https://github.com/sb2702/longpipe/blob/main/LICENSE) and [WEIGHTS_LICENSE](https://github.com/sb2702/longpipe/blob/main/WEIGHTS_LICENSE). You're free to self-host the weights inside other open-source or commercial projects under MIT-compatible terms. The optional MediaPipe selfie weights are Google's, under Apache-2.0 ([NOTICE](https://cdn.longpipe.dev/models/mediapipe/selfie/v1/NOTICE.md)).

The weights were trained on permissively-licensed public datasets (P3M-10k, AISegment, COCO, OpenImages), Z-Image synthetics, and a custom webcam dataset collected with participants' explicit, informed consent. The full dataset list and provenance details: [longpipe.dev/docs/licensing](https://longpipe.dev/docs/licensing).


This project is tested with BrowserStack
