# Longpipe 🐉

Fast, high quality virtual effects in the browser


<img width="640" height="360" alt="0087" src="https://github.com/user-attachments/assets/525f983c-6b3d-43ff-aedf-b1203e2955c1" />



Try the live demo: [longpipe.dev/demo](https://longpipe.dev/demo)

> **Warning** — This project is very new and still under active development. Expect API changes between versions and bugs.

## Features

- **Support across browsers:** Works on every browser, simplifying dozens of browser inconsistencies in one simple API.
- **Performance:** Longpipe was built from the ground up to work as well on 10-year-old netbooks as it does on the latest MacBook Pro.
- **Adaptivity:** Longpipe has several model variants (xl to xs), auto-selecting to provide the best quality while maintaining 30 fps.
- Built by the founder of [Vectorly](https://www.crunchbase.com/organization/vectorly), a commercial effects SDK acquired in 2021. A ground-up redesign for the WebGPU era, with years of production lessons baked in.

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


## Performance

Longpipe uses custom-trained models built with an EfficientNet-Lite encoder and a U-Net style decoder, broken into five performance presets (`xs` through `xl`) that vary in encoder size, decoder width, and input resolution. Across all variants Longpipe has much higher segmentation accuracy than alternative open-source models like MediaPipe and BodyPix, while also delivering much better real-world performance — the model runs as pure WebGPU/WebGL shaders in a zero-copy, fully-GPU pipeline.

### Quality
Using average alpha pixel error on the P3M-10K valdation dataset (499 landscape images), all variants of Longpipe surpassed mediapipe in both MAE and IoU.

<img width="640" alt="lp-quality" src="https://github.com/user-attachments/assets/c7044937-95af-496f-893a-a05a50d7c914" />

### Speed
With the pure GPU zero copy pipeline, Longpipe achieves better real world performance than mediapipe using much larger models.

<img width="640" alt="lp-speed" src="https://github.com/user-attachments/assets/ce658b8d-dc62-4d6b-90f8-cd62dfef2a59" />





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

// solid color — hex string or [r, g, b] floats in [0, 1]
new EffectsPipeline(stream, { background: { color: '#00b050' } })       // greenscreen
new EffectsPipeline(stream, { background: { color: [0, 0.7, 0.3] } })
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
  preset: { model: 'large', dtype: 'f32', resolution: { w: 640, h: 360 }, skipFrames: 0 },
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

By default Longpipe fetches model weights from `https://cdn.longpipe.dev/models/v/0.0.2/`. You can browse the available files, sizes, and SHA-256 hashes at [cdn.longpipe.dev/models/v/0.0.2/index.html](https://cdn.longpipe.dev/models/v/0.0.2/index.html) (machine-readable list at [manifest.json](https://cdn.longpipe.dev/models/v/0.0.2/manifest.json)).

To serve them yourself, mirror the files under any base URL with the same `model_${name}.bin` naming convention and pass it via `weightsBaseUrl`:

```ts
new EffectsPipeline(stream, {
  weightsBaseUrl: 'https://your-cdn.example.com/longpipe-weights/',
})
```

## Browser support

Works on Chromium (Chrome, Edge), Firefox, and Safari (desktop and iOS). WebGPU is used when available; WebGL2 is the fallback. Longpipe picks the optimal video frame transport for each browser internally — `MediaStreamTrackProcessor`, `transferControlToOffscreen` + `captureStream`, or an `ImageBitmap` shuttle as universal fallback — all invisible to the caller.



## How it works

Two layers:

- **Model** (`src/model/`) — an EfficientNet-Lite encoder (lite0; lite4 on the `xl` tier) with a U-Net decoder, plus a lightweight U-Net *wrapper* that sharpens the matte at higher resolution and a temporal ConvGRU that smooths it across frames. Written as TypeScript op classes — each layer is a class; weights load as binary tensors at init; the backend (WebGPU or WebGL2) is injected at construction. BatchNorm is fused into conv weights at export — there is no BN op at inference.
- **Pipeline** (`src/pipeline/`) — capability detection, per-browser frame transport selection, worker spawn, audio passthrough, autotune, and the adaptive controller. Designed to absorb the browser/codec/canvas plumbing complexity so consumers don't have to. 

Five trained presets cover the hardware range. "Resolution" is the model's working (canvas) resolution; the encoder runs at a lower internal resolution and the U-Net wrapper refines back up to this.

| Preset | Resolution | Encoder      | Decoder  | Skip frames |
|--------|------------|--------------|----------|-------------|
| xl     | 1280×720   | full (lite4) | 2× ch    | 0           |
| large  | 640×360    | full (lite0) | standard | 0           |
| medium | 512×288    | full (lite0) | standard | 1           |
| small  | 384×216    | small        | standard | 1           |
| xs     | 384×216    | small        | standard | 2           |

`Skip frames` is how many input frames the model sits out between runs — the compositor still renders every frame using the most recent alpha matte. Autotune picks one of these at init based on a microbenchmark of the actual network on the actual device.

## Development

Training scripts, fixture generation, and the weight export pipeline are not yet documented here — coming soon.


## Roadmap

- [x] Background segmentation / virtual backgrounds
- [x] WebGPU + WebGL2 backends, f16 + f32
- [x] Worker-based pipeline with per-browser transport selection
- [x] Autotuned + adaptive preset selection
- [ ] Face landmark detection
- [ ] Lighting correction
- [ ] Background noise removal (audio, separate pipeline)
- [ ] Mobile SDKs

## License

SDK source code is MIT-licensed — see [LICENSE](LICENSE).

The pre-trained model weights distributed via `cdn.longpipe.dev` (and any mirrored copies) are also MIT-licensed — see [WEIGHTS_LICENSE](WEIGHTS_LICENSE). You're free to self-host them inside other open-source or commercial projects under MIT-compatible terms.

### Training data and pretrained weights

Longpipe's pre-trained weights are released under MIT (see [WEIGHTS_LICENSE](WEIGHTS_LICENSE)). They were trained on:

- [P3M-10k](https://github.com/JizhiziLi/P3M) — MIT
- [AISegment Matting Human Datasets](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets) — MIT ([upstream license](https://github.com/aisegmentcn/matting_human_datasets/blob/master/LICENSE))
- [COCO](https://cocodataset.org) CC BY 4.0
- [OpenImages](https://storage.googleapis.com/openimages/web/index.html) CC by 4.0
- Synthetic images generated with [Z-Image](https://github.com/Tongyi-MAI/Z-Image) (text-to-image) — Apache 2.0
- A custom dataset of short webcam videos collected via [Prolific](https://www.prolific.com/), from participants who gave explicit, informed consent for their footage to be used to train an open-source virtual-background model. Faces are blurred before training; the raw videos are stored privately (EU), never distributed, and deleted after the training window.
- Pseudo-labels generated by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT), used as a teacher model

The encoder is initialized from a pre-trained [EfficientNet-Lite](https://github.com/RangiLyu/EfficientNet-Lite) backbone — Apache 2.0.

As is standard for trained models, the released weights are not a redistribution of any training image or video — they are published under MIT. Aside from COCO (whose underlying Flickr images carry mixed licenses), every source above is permissively licensed or used with explicit consent. If your use case has strict data-provenance requirements, review COCO's image terms for yourself.
