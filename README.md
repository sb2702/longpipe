# Longpipe 🐉

Fast, high quality virtual effects in the browser


<img width="960" height="540" alt="video-effects-sdk" src="https://github.com/user-attachments/assets/5dbe1ccb-09dd-4f22-b4be-e3122767dbcc" />




Try the live demo: [longpipe.dev/demo](https://longpipe.dev/demo)

> **Warning** — This project is very new and still under active development. Expect API changes between versions and bugs.

## Features

- **Support across browsers:** Works on every browser, simplifying dozens of browser inconsistencies in one simple API.
- **Performance:** Longpipe was built from the ground up to work as well on 10-year-old netbooks as it does on the latest MacBook Pro.
- **Adaptivity:** Longpipe has several model variants (xl to xs), auto-selecting to provide the best quality while maintaining 30 fps.
- **Effects that compose:** virtual backgrounds, skin touch-up, and auto-reframe all run together off a single shared encoder pass — enabling one doesn't cost you another inference.
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
  audio: 'denoise'
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

`'auto'` runs a microbenchmark at init and picks the largest preset that fits the per-frame budget on the current device, **topping out at `large`** — `xl` is the explicit opt-in flagship (`'quality'` or a manual config). While `'auto'` is in effect an adaptive controller polls FPS / model time and swaps preset up or down as conditions change. Explicit preset choices (`'fast'`, `'balanced'`, `'quality'`, or a manual config) are always respected and never auto-overridden.

You can also pass a manual config:

```ts
new EffectsPipeline(stream, {
  preset: { model: 'large', dtype: 'f32', resolution: { w: 640, h: 400 }, skipFrames: 0 },
})
```

## Audio denoise

Real-time speech denoising runs as a **separate AudioWorklet pipeline** on the audio render thread — independent of the video/GPU pipeline, in parallel. Pass `audio: 'denoise'` and the input stream's mic is cleaned in place; `pipeline.stream` carries the denoised audio track.

```ts
const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true })
new EffectsPipeline(stream, { audio: 'denoise' })   // auto-picks a model for the device
```

Three models back three tiers. `auto` (the default) runs a tiny **weight-free probe** at init — it times the real network on this device *without* downloading the multi-MB weights — and picks the best fit:

| `model`   | tier | what it is |
|-----------|------|------------|
| `dfn`     | high | DeepFilterNet3, full f32 — best quality |
| `dfnint8` | mid  | DeepFilterNet3 with int8 GRUs — smaller download, faster on weak hardware |
| `rnnoise` | low  | classic RNNoise — tiny, and the fallback where wasm SIMD is unavailable |

```ts
// force a specific model or tier instead of probing
new EffectsPipeline(stream, { audio: { denoise: { model: 'dfn' } } })

interface DenoiseOptions {
  model?:          'auto' | 'high' | 'mid' | 'low' | 'rnnoise' | 'dfn' | 'dfnint8'  // default 'auto'
  postFilterBeta?: number    // DFN post-filter, suppresses residual noise (default 0.03)
  gruLeak?:        number     // DFN recurrent-drift bound (default 0.995)
  enabled?:        boolean    // start denoising vs. passthrough (default true)
}
```

Toggle, reconfigure, and inspect at runtime:

```ts
pipeline.setDenoise(false)                      // passthrough (cheap to re-enable)
pipeline.setDenoise({ postFilterBeta: 0.05 })   // tweak DFN params live
pipeline.getAudioStats()                        // { model, p50Ms, p95Ms, latencyMs, sampleRate, active }
```

The denoiser handles sample-rate conversion internally (it runs at 48 kHz and resamples when the device can't). `pipeline.ready` resolves on the video first frame and does **not** wait on audio — denoising joins asynchronously, with the mic passing through until it's live.

## Other options

```ts
new EffectsPipeline(stream, {
  background:       'blur',
  touchup:          { strength: 0.6 },             // skin smoothing; omit to disable
  reframe:          true,                          // auto-frame the subject; omit to disable
  preset:           'auto',
  adaptive:         true,                          // default; only applies when preset is 'auto'
  audio:            'denoise',                 // 'passthrough' | 'drop' | 'denoise' | { denoise: {...} }
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

By default Longpipe fetches model weights from `https://cdn.longpipe.dev/models/v/0.0.5/`. You can browse the available files, sizes, and SHA-256 hashes at [cdn.longpipe.dev/models/v/0.0.5/index.html](https://cdn.longpipe.dev/models/v/0.0.5/index.html) (machine-readable list at [manifest.json](https://cdn.longpipe.dev/models/v/0.0.5/manifest.json)).

To serve them yourself, mirror the files under any base URL with the same `model_${name}.bin` naming convention and pass it via `weightsBaseUrl`:

```ts
new EffectsPipeline(stream, {
  weightsBaseUrl: 'https://your-cdn.example.com/longpipe-weights/',
})
```

If you use touch-up, three more files must sit alongside the tier weights — they're fetched from the same base URL the first time the effect is enabled: `model_landmark_mesh.bin` (and `.f16.bin`), `face_topology.json`, and `weight_mask.png`. Miss these and matting still works, but touch-up fails to enable. Auto-reframe needs nothing extra.

## Browser support

Works on Chromium (Chrome, Edge), Firefox, and Safari (desktop and iOS). WebGPU is used when available; WebGL2 is the fallback. Longpipe picks the optimal video frame transport for each browser internally — `MediaStreamTrackProcessor`, `transferControlToOffscreen` + `captureStream`, or an `ImageBitmap` shuttle as universal fallback — all invisible to the caller.



## Face touch-up

UV-space skin smoothing (the "beautify" filter). It runs **in parallel with any background effect** — both features share one encoder pass per frame, and the retouched frame feeds the background compositor.

```ts
new EffectsPipeline(stream, {
  background: 'blur',
  touchup: { strength: 0.6 },   // presence enables it
})

await pipeline.setTouchup({ strength: 0.8, style: 'bilateral' })   // live update
await pipeline.setTouchup(null)                                    // disable
```

```ts
interface TouchupOptions {
  strength?: number   // 0..1 — blend of smoothed skin over original (default 0.6)
  amount?:   number   // smoothing radius (default 8)
  detail?:   number   // high-frequency (pore texture) keep — freq-sep only (default 0.35)
  style?:    'freq-sep' | 'bilateral'   // default 'freq-sep'
}
```

How it works: a 5-point face-keypoint head rides the matting encoder (no second encoder pass); its heatmaps are decoded on the GPU into a face box; a 478-point landmark model runs on the face crop; the face is unwrapped into a pose-independent UV atlas, smoothed there, and composited back through a per-region weight mask. Eyes, brows, and lips are protected and stay pixel-exact; `'freq-sep'` smooths while keeping pore-level texture, `'bilateral'` is a stronger edge-preserving filter. The whole chain is GPU-resident — no per-frame readback.

On first enable the landmark weights (`model_landmark_mesh.bin`, ~2.6 MB) and two small static assets are fetched from `weightsBaseUrl`, then cached; parameter updates are instant. With no face in frame the effect passes through cleanly.

Everyone in frame gets smoothed — up to **four faces** on the `medium`, `large` and `xl` presets. (`small` and `xs` decode keypoints on a coarser grid that can't reliably separate a second face, so they smooth one.) There's no per-face API and nothing to configure; the faces share a single atlas, so the smoothing passes run once regardless of how many people are in shot.

## Auto-reframe

Crop and zoom the output so the subject is framed — the "centre stage" behaviour from Meet, Zoom and Teams. It rides the same face keypoints as touch-up, so it needs **no extra model and no extra assets**: turning it on costs a crop, not an inference.

```ts
new EffectsPipeline(stream, { reframe: true })                 // auto-follow, all defaults
new EffectsPipeline(stream, { reframe: { auto: false } })      // solve once, then freeze

await pipeline.setReframe({ zoom: 1.6 })   // enable / update live
await pipeline.reframe()                   // manual mode: re-solve now
await pipeline.setReframe(null)            // disable
```

```ts
type ReframeConfig = boolean | ReframeOptions

interface ReframeOptions {
  zoom?:    number   // crop = frame / zoom (default 1.35; relaxes toward 1 as needed)
  gravity?: number   // 0..1 pull toward the subject (default 0.5; 1 would centre it exactly)
  margin?:  number   // keep-out space around the face (default 0.04)
  auto?:    boolean | { deadband?: number; ease?: number }   // default true
}
```

Two modes, because apps ship both. **auto** (default) follows the subject with a deadband — it holds still, makes a deliberate move when they've actually gone somewhere, then holds again; continuous tracking reads as swimming, because a head moves constantly even when a person doesn't. **manual** (`auto: false`) frames once and freezes until you call `pipeline.reframe()` — the mode for a reframe button in your own UI.

The crop must stay inside the frame *and* contain the subject, which has a useful consequence: if you sit hard in a corner, auto-reframe does nothing at all, because containing you would need a crop the size of the whole frame. That falls out of the constraints rather than being a special case, and it matches what the commercial implementations do.

It reframes **you**, not your background: with a virtual background the image stays put while you zoom (it's a backdrop behind you, not part of the shot), while a *blurred* background reframes with you, since that is your real room. Neither needs configuring. Auto-reframe frames one subject — the largest face — and the choice is sticky, so two people side by side won't make the camera flip back and forth. That's the opposite of touch-up, which smooths everyone: framing is a choice about whose shot it is; smoothing isn't.

## How it works

Two layers:

- **Model** (`src/model/`) — a shared EfficientNet-Lite encoder (lite0; lite3 on the `xl` tier) feeding multiple heads: a U-Net matting decoder (sharpened at higher resolution by a lightweight U-Net *wrapper*), a 5-point face-keypoint head (drives touch-up and auto-reframe), and a small optical-flow head that provides temporal stability — the renderer warps the previous outputs along the predicted flow and blends them through a flow-gated stabilizer, so the matte is steady without any recurrent state in the network. A separate 478-point landmark model runs on the detected face crop. Written as TypeScript op classes — each layer is a class; weights load as binary tensors at init; the backend (WebGPU or WebGL2) is injected at construction. BatchNorm is fused into conv weights at export — there is no BN op at inference.
- **Pipeline** (`src/pipeline/`) — capability detection, per-browser frame transport selection, worker spawn, audio passthrough, autotune, and the adaptive controller. Designed to absorb the browser/codec/canvas plumbing complexity so consumers don't have to. 

Five trained presets cover the hardware range. "Resolution" is the model's working (canvas) resolution; the encoder runs at a lower internal resolution and the U-Net wrapper refines back up to this.

| Preset | Resolution | Encoder      | Decoder  | Skip frames |
|--------|------------|--------------|----------|-------------|
| xl     | 1280×768   | full (lite3) | 2× ch    | 0           |
| large  | 640×400    | full (lite0) | standard | 0           |
| medium | 512×320    | full (lite0) | standard | 1           |
| small  | 384×224    | small        | standard | 1           |
| xs     | 384×240    | small        | standard | 2           |

`Skip frames` is how many input frames the model sits out between runs — the compositor still renders every frame using the most recent alpha matte. Autotune picks one of these at init based on a microbenchmark of the actual network on the actual device.

## Development

Training scripts, fixture generation, and the weight export pipeline are not yet documented here — coming soon.


## Roadmap

- [x] Background segmentation / virtual backgrounds
- [x] Background noise removal (audio, separate pipeline)
- [x] Face landmarks + touch-up
- [x] Multi-face support
- [x] Auto-reframe
- [ ] AR effects
- [ ] Lighting correction


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

As is standard for trained models, the released weights are not a redistribution of any training image or video — they are published under MIT.
