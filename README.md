# Longpipe 🐲

Hardware-accelerated real-time AI media processing in the browser.

Try the live demo: [longpipe.dev/demo](https://longpipe.dev/demo)

> **Warning** — This project is very new (v0.0.1) and still under active development. Expect API changes between versions and bugs.

## Features

- Real-time portrait matting / virtual backgrounds in the browser
- WebGPU compute shaders with a WebGL2 fragment-shader fallback
- Worker-by-default — all model and render work runs off the main thread
- Background options: blur, image, video, solid color, or none (passthrough)
- Six trained model presets covering hardware from MacBook Pro down to Chromebook
- Auto-tuned at init: picks the best preset for the current device, plus the optimal frame transport for the current browser
- Adaptive at runtime: swaps to a smaller preset when FPS drops, larger when there's headroom
- Single-call API: `new EffectsPipeline(inputStream, { background })` returns a `MediaStream`
- Audio passthrough by default

## Install

```
npm install longpipe
```

## Quick start

```ts
import { EffectsPipeline } from 'longpipe'

const inputStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })

const pipeline = new EffectsPipeline(inputStream, {
  background: 'blur',
})

videoEl.srcObject = pipeline.stream   // available immediately
await pipeline.ready                  // optional — resolves once the effect is live
```

`pipeline.stream` is wired synchronously and emits the unprocessed input until the model is ready (~1–3s on cold start, depending on hardware), so the consumer sees live video the whole time.

## Backgrounds

The `background` option takes a wide set of inputs and normalizes them internally:

```ts
new EffectsPipeline(stream, { background: 'none' })                 // passthrough
new EffectsPipeline(stream, { background: 'blur' })                 // default sigma
new EffectsPipeline(stream, { background: { blur: { sigma: 12 } } })

// images: URL, <img>, ImageBitmap, or { image: ... }
new EffectsPipeline(stream, { background: 'https://example.com/bg.jpg' })
new EffectsPipeline(stream, { background: imgElement })
new EffectsPipeline(stream, { background: imageBitmap })

// video: URL, Blob, or <video> — looped, muted, decoded on the main thread
new EffectsPipeline(stream, { background: { video: 'https://example.com/bg.mp4' } })

// solid color
new EffectsPipeline(stream, { background: { color: [0, 100, 200] } })
```

Swap at runtime:

```ts
await pipeline.setBackground({ blur: { sigma: 20 } })
await pipeline.setBackground('https://example.com/other-bg.jpg')
```

## Performance presets

```ts
new EffectsPipeline(stream, { preset: 'auto' })       // benchmarks at init (default)
new EffectsPipeline(stream, { preset: 'fast' })       // small model
new EffectsPipeline(stream, { preset: 'balanced' })   // medium model
new EffectsPipeline(stream, { preset: 'quality' })    // xl model
```

`'auto'` runs a microbenchmark at init and picks the largest preset that fits the per-frame budget on the current device. While `'auto'` is in effect an adaptive controller polls FPS / model time and swaps preset up or down as conditions change. Explicit preset choices (`'fast'`, `'balanced'`, `'quality'`, or a manual config) are always respected and never auto-overridden.

You can also pass a manual config:

```ts
new EffectsPipeline(stream, {
  preset: { model: 'large', dtype: 'f16', resolution: { w: 256, h: 144 }, skipFrames: 0 },
})
```

## Other options

```ts
new EffectsPipeline(stream, {
  background:       'blur',
  preset:           'auto',
  adaptive:         true,                          // default; only applies when preset is 'auto'
  audio:            'passthrough',                 // or 'drop'
  outputResolution: { w: 1280, h: 720 },           // default: matches the input video track
  weightsBaseUrl:   'https://your-cdn/longpipe/',  // default: cdn.longpipe.dev
  enabled:          true,                          // false = pass input through unchanged
  onReady:          () => console.log('live'),
  onError:          (err) => console.error(err),
})
```

Toggle the effect on/off without tearing anything down (cheap to re-enable):

```ts
pipeline.setEnabled(false)
```

Tear it all down:

```ts
pipeline.destroy()
```

## Self-hosting weights

By default Longpipe fetches model weights from `https://cdn.longpipe.dev/models/v/0.0.1/`. You can browse the available files, sizes, and SHA-256 hashes at [cdn.longpipe.dev/models/v/0.0.1/index.html](https://cdn.longpipe.dev/models/v/0.0.1/index.html) (machine-readable list at [manifest.json](https://cdn.longpipe.dev/models/v/0.0.1/manifest.json)).

To serve them yourself, mirror the files under any base URL with the same `model_${name}.bin` naming convention and pass it via `weightsBaseUrl`:

```ts
new EffectsPipeline(stream, {
  weightsBaseUrl: 'https://your-cdn.example.com/longpipe-weights/',
})
```

## Browser support

Works on Chromium (Chrome, Edge), Firefox, and Safari (desktop and iOS). WebGPU is used when available; WebGL2 is the fallback. Longpipe picks the optimal video frame transport for each browser internally — `MediaStreamTrackProcessor`, `transferControlToOffscreen` + `captureStream`, or an `ImageBitmap` shuttle as universal fallback — all invisible to the caller.

## Problem

If you want real-time AI media processing in the browser — virtual backgrounds, background noise removal, face filters — one of the few open-source options is [MediaPipe](https://github.com/google-ai-edge/mediapipe). MediaPipe ships some great open-source models, but they all run in WebAssembly, and using them effectively still requires a fair amount of hardware-accelerated pre- and post-processing on top.

![MediaPipe pipeline diagram](https://github.com/user-attachments/assets/62932072-c5b2-445e-b5d8-a2bd5bb72920)

In 2021, I built an SDK to implement [hardware-accelerated neural networks directly in WebGL](https://medium.com/vectorly/building-a-more-efficient-background-segmentation-model-than-google-74ecd17392d5) that proved much more efficient than the MediaPipe implementation. It was popular, but it was a commercial SDK — the company was acquired and the technology was never opened up.

In 2022, Google Meet adopted a similar approach with their own [hardware-accelerated networks](https://research.google/blog/high-definition-segmentation-in-google-meet/), but that was never open-sourced either.

As of late 2025, MediaPipe is still the state of the art for open-source real-time browser ML, and nothing better has come along.

## Solution

With WebGPU shipping and WebNN on the way, it is now possible to build efficient implementations of popular real-time media-processing features — background segmentation, audio filtering — by combining hardware-accelerated neural networks in WebGPU/WebNN with efficient pre- and post-processing, so developers can plug in features like virtual backgrounds and noise removal without worrying about the details.

Longpipe is that SDK.

## How it works

Two layers:

- **Model** (`src/model/`) — EfficientNet-Lite-0 encoder with a U-Net decoder, written as TypeScript op classes. Each layer is a class; weights load as binary tensors at init; the backend (WebGPU or WebGL2) is injected at construction. BatchNorm is fused into conv weights at export — there is no BN op at inference. See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) and [docs/MODEL_PLAN.md](../docs/MODEL_PLAN.md).
- **Pipeline** (`src/pipeline/`) — capability detection, per-browser frame transport selection, worker spawn, audio passthrough, autotune, and the adaptive controller. Designed to absorb the browser/codec/canvas plumbing complexity so consumers don't have to. See [docs/PIPELINE.md](../docs/PIPELINE.md).

Six trained presets cover the hardware range:

| Preset  | Resolution | Encoder | Decoder  | Skip frames |
|---------|------------|---------|----------|-------------|
| xl      | 512×288    | full    | 2× ch    | 0           |
| large   | 256×144    | full    | standard | 0           |
| medium  | 256×144    | full    | standard | 1           |
| compact | 256×144    | full    | small    | 1           |
| small   | 256×144    | small   | standard | 1           |
| xs      | 192×108    | small   | standard | 1           |

`Skip frames` is how many input frames the model sits out between runs — the compositor still renders every frame using the most recent alpha matte. Autotune picks one of these at init based on a microbenchmark of the actual network on the actual device.

## Development

Training scripts, fixture generation, and the weight export pipeline are not yet documented here — coming soon.

## Acknowledgements

- [EfficientNet-Lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite) for the backbone architecture.
- [Google Meet HD Segmentation research](https://research.google/blog/high-definition-segmentation-in-google-meet/) for the architectural choices that hold up on accelerators.
- [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) for making compute-heavy work in the browser realistic.
- [Mediabunny](https://mediabunny.dev/) — a colleague's project and a model for what an open-source media SDK with sponsorship-backed development can look like.

## Roadmap

- [x] Background segmentation / virtual backgrounds
- [x] WebGPU + WebGL2 backends, f16 + f32
- [x] Worker-based pipeline with per-browser transport selection
- [x] Autotuned + adaptive preset selection
- [ ] Face landmark detection
- [ ] Lighting correction
- [ ] Background noise removal (audio, separate pipeline)
- [ ] Mobile SDKs
