# Benchmark methodology

What the Longpipe vs MediaPipe benchmarks measure, how, what they can't tell you,
and how they relate to the numbers previously shown in the README. Commands to
re-run everything are in [`README.md`](README.md).

Two independent benchmarks:

- **Quality** ([`quality/`](quality/)): alpha-matte accuracy on the P3M-10k
  validation sets, in Python (ONNX Runtime + the official MediaPipe Python API).
- **Performance** ([`performance/`](performance/)): speed in the browser, using the
  real SDK model layer (`longpipe/model`) and MediaPipe Tasks for Web, in a
  WebCodecs decode → background replacement → H.264 encode pipeline.

They run the same model weights (the ONNX exports are the SDK's weights, and the
MediaPipe `.tflite` files are the same in both), but the quality numbers come from
fp32 ONNX Runtime on a CPU, not from the browser. See [Caveats](#caveats).

---

## Quality

### Data

**P3M-10k** (Li et al., *Privacy-Preserving Portrait Matting*, ACM MM 2021,
[github.com/JizhiziLi/P3M](https://github.com/JizhiziLi/P3M)): its two 500-image
validation sets with human-labelled alpha mattes and trimaps:

| Set | Images | What |
|---|---|---|
| P3M-500-P | 500 | Face-blurred portraits (the same anonymisation as the training split) |
| P3M-500-NP | 500 | Un-blurred celebrity portraits, a different source from the training split |

The dataset is distributed under a release agreement and is **not** included in
this repo (no images, not even examples). The scripts take a path to your own
download; see [`README.md`](README.md#getting-p3m-10k). Only aggregate and
per-image metrics (filename + numbers) are committed.

Orientation (w/h > 1.05 landscape, < 0.95 portrait, otherwise square): 499
landscape, 466 portrait, 35 square across the 1,000 images.

### Methods

| Method | Model | Input | Native mask (W×H) |
|---|---|---|---|
| `longpipe-xs` | `onnx/xs/matting_{landscape,portrait}.onnx` | 384×240 | 384×240 |
| `longpipe-small` | `onnx/small/…` | 384×224 | 384×224 |
| `longpipe-medium` | `onnx/medium/…` | 512×320 | 512×320 |
| `longpipe-large` | `onnx/large/…` | 640×400 | 640×400 |
| `longpipe-xl` | `onnx/xl/…` | 1280×768 | 1280×768 |
| `mediapipe-square` | `selfie_segmenter.tflite` (float16/1) | 256×256 | 256×256 |
| `mediapipe-landscape` | `selfie_segmenter_landscape.tflite` (float16/1) | 256×144 | 256×144 |

Portrait images use the portrait Longpipe export (W and H swapped). Sizes are the
landscape canvas sizes of `TIER_CONFIG` in `src/model/tier_config.ts`.

- **Longpipe**: the official ONNX exports from
  [huggingface.co/Longpipe/longpipe](https://huggingface.co/Longpipe/longpipe),
  downloaded with `huggingface_hub` at a pinned revision (`521aefe…`, recorded in
  the results JSON with every file's sha256). These are the SDK's weights (the
  model card documents the export). The image is resized to the canvas size with
  plain bilinear sampling (half-pixel centres, no antialiasing: the same sampling
  as the SDK's `Input` op, a GPU bilinear texture fetch), and the output alpha is
  bilinearly upsampled to the ground-truth size (as the SDK's `RenderOp` does).
  The image is stretched, not cropped, to the canvas aspect ratio, as the SDK's
  `Input` op does.
- **MediaPipe**: the official `mediapipe` Python package, Tasks API
  (`vision.ImageSegmenter`, `RunningMode.IMAGE`, `output_confidence_masks=True`).
  The full-resolution image goes in; MediaPipe does its own resize to the model
  input, and returns the confidence mask already resampled to the input image
  size. That mask is used as the alpha matte. Model files are downloaded from
  Google's model bucket and checked by sha256:
  - `selfie_segmenter.tflite` (square): the default model in MediaPipe's image
    segmentation docs and the model used by the
    [webrtcHacks article benchmark](https://github.com/sb2702/webrtc-hacks-webgpu/tree/main/segmentation),
    `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite`,
    sha256 `191ac952…658b`.
  - `selfie_segmenter_landscape.tflite`: the variant for 16:9 video,
    `…/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite`,
    sha256 `490e9ea7…6b8e` (the `latest` URL serves the same file, and it is
    byte-identical to the file used for the old README chart).
  - Native input/output shapes were checked with the TFLite interpreter:
    `[1,256,256,3] → [1,256,256,1]` and `[1,144,256,3] → [1,144,256,1]`.

  Both are applied to every image whatever its orientation, as a user of either
  model would. On portrait images the landscape model is a poor fit (it squashes
  a tall image into 256×144); the square model is the fairer MediaPipe baseline
  for the full set, and both are reported.

### Metrics

All at the ground-truth resolution (the original image size; images average
~1.7 MP), alpha in [0, 1], averaged over images. The definitions and scale
factors are P3M's own (`core/evaluate.py` in the P3M repo, used for the paper's
tables). `eval_p3m.py` reimplements them without PyTorch / scikit-image; on real
masks it matches the reference code exactly (checked on sample images during
development):

| Metric | Definition |
|---|---|
| **MAD** | mean \|pred − gt\| over the image (= MAE) |
| **MSE** | mean (pred − gt)² over the image |
| **SAD** | Σ \|pred − gt\| / 1000 |
| **Grad** | Σ (‖∇pred‖ − ‖∇gt‖)² / 10, Gaussian-derivative gradients, σ = 1.4 |
| **Conn** | P3M's connectivity error (threshold step 0.1), / 1000 |
| **IoU@0.5** | IoU of `pred ≥ 0.5` and `gt ≥ 0.5` (not a P3M metric; added because MediaPipe is a segmentation model) |
| **MAD / SAD (trimap unknown)** | P3M's trimap variant: pixels in the trimap's known foreground / background are set to 1 / 0, so only the unknown band (hair, edges) counts; MAD is normalised by the number of unknown pixels |

Lower is better for all except IoU.

Two protocols:

- **`standard`** (the headline): all 1,000 images, full frame, all metrics,
  per set and overall, plus landscape-only and portrait-only breakdowns.
- **`legacy-readme`**: reproduces the protocol behind the old README chart (see
  [below](#relation-to-the-old-readme-chart)): the 499 landscape images only,
  each centre-cropped to 16:9, MAD and IoU only, MediaPipe landscape model.

Everything is deterministic (no augmentation, no randomness; ONNX Runtime CPU,
fp32, one thread per worker process).

### Results (2026-11-15)

Full tables: [`quality/results/2026-11-15-p3m.md`](quality/results/2026-11-15-p3m.md).

| Method | P3M-500-P MAD | P3M-500-P IoU | P3M-500-NP MAD | P3M-500-NP IoU | All MAD | All IoU | All Grad | All MAD (unknown) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Longpipe xs | 0.0210 | 0.9229 | 0.0233 | 0.9262 | 0.0221 | 0.9246 | 41.0 | 0.151 |
| Longpipe small | 0.0170 | 0.9365 | 0.0155 | 0.9536 | 0.0162 | 0.9451 | 35.4 | 0.120 |
| Longpipe medium | 0.0106 | 0.9657 | 0.0102 | 0.9716 | 0.0104 | 0.9687 | 28.9 | 0.091 |
| Longpipe large | 0.0111 | 0.9632 | 0.0100 | 0.9720 | 0.0106 | 0.9676 | 27.1 | 0.088 |
| Longpipe xl | **0.0071** | **0.9770** | **0.0080** | **0.9787** | **0.0075** | **0.9779** | **21.7** | **0.065** |
| MediaPipe selfie (square) | 0.0304 | 0.9228 | 0.0249 | 0.9398 | 0.0277 | 0.9313 | 36.4 | 0.150 |
| MediaPipe selfie (landscape) | 0.0322 | 0.9173 | 0.0276 | 0.9340 | 0.0299 | 0.9256 | 39.0 | 0.167 |

What this shows:

- On whole-image error (MAD, MSE, SAD, Conn), every Longpipe preset has lower
  error than both MediaPipe models on both sets. xl's MAD is ~3.7× lower than
  MediaPipe square's.
- **IoU@0.5 is not a clean sweep.** MediaPipe square has a higher IoU than
  Longpipe **xs** overall (0.931 vs 0.925) and on P3M-500-NP (0.940 vs 0.926);
  on P3M-500-P they tie (0.9228 vs 0.9229). small and up beat it.
- **Edge quality at the small presets is no better than MediaPipe's.** On Grad,
  MediaPipe square (36.4) beats xs (41.0) and is level with small (35.4); in the
  trimap unknown band, xs and MediaPipe square are level (0.151 vs 0.150 MAD).
  medium and up are clearly better on both.
- **Portrait images** (466): MediaPipe square beats xs on MAD (0.0249 vs 0.0261)
  and IoU (0.9405 vs 0.9235), and is just behind small on IoU (0.9449). Every
  Longpipe preset is worse on portrait than on landscape images; MediaPipe square
  is the other way round.
- medium and large are within noise of each other on whole-image metrics (large
  is better on Grad and the unknown band); large is not uniformly better.

### Relation to the old README chart

An earlier README chart (May 2026 models, private harness) used only the 499
landscape images of P3M-500-P/NP, centre-cropped to 16:9, against MediaPipe's
landscape model. `--protocol legacy-readme` reproduces it: MediaPipe's numbers match
exactly (0.0306 MAD / 0.919 IoU), the shipped xl–small match or beat the old chart,
and the shipped xs is worse than the chart's xs (0.0176 vs 0.0139 MAD). That protocol
wasn't disclosed and favours Longpipe (it drops the portrait images where MediaPipe's
square model does best), so the standard protocol above is the headline.

---|---|---|---|
| | MAD / IoU | MAD / IoU | MAD / IoU |
| xl | 0.0077 / 0.979 | 0.0062 / 0.979 | 0.0075 / 0.978 |
| large | 0.0108 / 0.969 | 0.0087 / 0.974 | 0.0106 / 0.968 |
| medium | 0.0108 / 0.969 | 0.0104 / 0.967 | 0.0104 / 0.969 |
| small | 0.0125 / 0.963 | 0.0120 / 0.957 | 0.0162 / 0.945 |
| xs | 0.0139 / 0.960 | 0.0176 / 0.933 | 0.0221 / 0.925 |
| MediaPipe landscape | 0.0306 / 0.919 | **0.0306 / 0.919** | 0.0299 / 0.926 |
| MediaPipe square | – | – | 0.0277 / 0.931 |

- The MediaPipe row reproduces exactly (0.03065 / 0.9186 then and now; same model
  file), so the protocol is reproduced faithfully and this harness agrees with
  the old one.
- On that protocol, the shipped xl, large, medium and small have about the same or
  lower MAD than the chart; **the shipped xs is worse than the chart's xs** (MAD
  0.0176 vs 0.0139, IoU 0.933 vs 0.960). The chart overstates today's xs.
- On all 1,000 images (portraits included), Longpipe's numbers get worse and
  MediaPipe's get better (the square model in particular), which is why xs loses
  to MediaPipe on IoU in the standard protocol. "Every Longpipe variant beats
  MediaPipe on both MAD and IoU" holds for the landscape-only protocol; under the
  standard protocol it holds for MAD and for small and up on IoU, but not for xs.
- The old protocol is defensible for webcam use (landscape video) but it was not
  disclosed, and it favours Longpipe: it drops the image orientation where
  MediaPipe square does best and Longpipe worst.

---

## Performance

### What runs: model × engine × pipeline

Every pipeline has the same interface, `process(VideoFrame) → VideoFrame`, and
does the same effect: replace the background with solid green,
`mix(green, frame, alpha)`, with the alpha bilinearly upsampled to the frame size,
on the GPU, output as a new `VideoFrame` at the input resolution. Three things
vary, and the rows are organised by them:

- **Model**: MediaPipe's selfie segmenter (square 256×256 or landscape 256×144;
  the same `.tflite` files as the quality benchmark) or Longpipe's own network
  (presets xs–xl).
- **Engine**: what runs the model: MediaPipe Tasks for Web, LiteRT.js (Google's
  TFLite web runtime), or Longpipe's WebGPU / WebGL2 backends.
- **Pipeline**: the network on every frame, or Longpipe's shipped per-frame
  schedule (key frames + learned optical-flow frames + temporal stabiliser).

| Row | Model | Engine | Pipeline | Code |
|---|---|---|---|---|
| 1 | MediaPipe selfie (square, landscape) | `@mediapipe/tasks-vision` 1.1.1 `ImageSegmenter`, GPU delegate (WebGL) and CPU delegate (WASM/XNNPACK) | every frame | MediaPipe's own resize; mask returned at frame size; GPU: composited by our WebGL shader in MediaPipe's context (no readback); CPU: mask uploaded, composited by our WebGPU shader |
| 2 | MediaPipe selfie (square, landscape) | `@litertjs/core` 2.5.3, WebGPU accelerator, default precision (fp32) | every frame | Ported from the webrtcHacks article harness: our WebGPU compute pass resizes the frame into LiteRT's input buffer (4-tap box filter), LiteRT runs on our `GPUDevice`, its output buffer is composited by our WebGPU shader; no readback |
| 3 | MediaPipe selfie (square, landscape) | Longpipe 0.0.17 `MediaPipeSelfie` (`longpipe/model`), WebGPU and WebGL2, f16 | every frame | SDK `Input` op → network → `RenderOp` (bilinear upscale, solid composite): exactly what `EffectsPipeline({ model: 'mediapipe-selfie' })` runs per frame (`Renderer.setMediaPipeSelfie`, `src/pipeline/worker/renderer.ts:601`) |
| 4 | Longpipe xs–xl | Longpipe 0.0.17 `TierModel` (`longpipe/model`), WebGPU and WebGL2, f16 (+ f32 large/xl) | **network every frame** | SDK `Input` op → `TierModel` → `RenderOp`; no flow path, no stabiliser |
| 5 | Longpipe xs–xl | Longpipe 0.0.17 `Renderer` (the SDK worker's per-frame core), WebGPU and WebGL2, f16 | **as shipped** | The SDK's own `Renderer.process(frame)` with a fixed preset; see below |
| – | none | – | passthrough / canvas copy | Ceilings: decoded frame straight to the output stage; or the frame drawn through the same composite shaders with alpha = 1 (WebGPU, WebGL2) |

Builds measured: rows 3–4 use the npm package `longpipe@0.0.17`. Row 5 drives
`Renderer` (`src/pipeline/worker/renderer.ts`), which the package doesn't export
(it is bundled into the inlined worker), so it is compiled by Vite from this
repo's `src/` at the 0.0.17 release commit (`76dc675`; `src/` unchanged on the
benchmarks branch, recorded in the results JSON as `sdkSource`), with the
backends from the same source. Same code as the package, different bundler
(esbuild/Vite vs tsup), unminified.

**Precision.** `EffectsPipeline` creates one backend for every preset with
`dtype: 'f16'` by default (`src/pipeline/index.ts:397`), falling back to f32
only when WebGPU lacks `shader-f16` (`src/pipeline/worker/setup_backend.ts:61`;
WebGL2 takes the dtype as given, `:92`), and patches the preset's dtype to the
backend's (`src/pipeline/worker/resolve_preset.ts:34`). So the f32 listed for
large/xl in `src/pipeline/presets.ts` is **not** what runs by default: as
shipped, every preset runs f16 on the M4, with the `.f16.bin` weights
(`fetchModelWeights`, `src/pipeline/index.ts:207`). Rows 3–5 use f16 for that
reason; row 4 also has f32 large/xl, the precision the first 2026-11-15 run used.
MediaPipe's own precision isn't exposed by its API; LiteRT runs at its default
(fp32; the article harness found fp16 the same speed on this machine).

**Not included anywhere**: the SDK's worker thread and `MediaStreamTrack`
transports (all pipelines run on the page's main thread, fed by the same
decoder), autotune and the adaptive controller (a fixed preset disables both:
`src/pipeline/index.ts:468`, `resolve_preset.ts`), touch-up and reframe (off).

### Longpipe as shipped: what runs on each frame

`EffectsPipeline` in Chrome: the worker reads camera frames, and its output
adapter calls `renderer.process(frame)` and wraps the renderer's canvas in a new
`VideoFrame` (`src/pipeline/worker/adapters/output_mstg.ts:15-16`). Row 5 does
exactly that, with the decoder as the source. With a background effect on,
`Renderer.process` (`renderer.ts:273`) takes the flow path whenever the loaded
weights carry a `flow` blob (`renderer.ts:289`, wired in `setPreset`,
`renderer.ts:653-683`). **All five shipped `.bin` files carry one** (checked
with `loadWeightsFromBinary`: `base, wrapper, flow, face`), so every preset
runs `Renderer.stepFlow` (`renderer.ts:382-420`):

**Off frames are not skipped frames.** Which frames are key frames is set by the
preset's `skipFrames` (`src/pipeline/presets.ts:30-34`; `shouldRunModel`,
`renderer.ts:532-539`): **xs 2** (key frame every 3rd frame), **small and
medium 1** (every 2nd), **large and xl 0** (every frame). Per frame:

| Step | key frame, skip presets (xs, small, medium) | off frame, skip presets | every frame, large / xl |
|---|---|---|---|
| Frame → network input (canvas res), → base res | ✓ | ✓ | ✓ |
| **Full matting network** (`TierModel`: encoder + U-Net wrapper) | ✓ | – | ✓ |
| **Optical-flow network** (`OpticalFlowNet`: a small learned net on the base-res frame pair, fused with the last key frame's cached encoder taps; `src/model/networks/optical_flow_net.ts`) | – (reuses the last flow) | ✓ last key frame → current | ✓ previous → current, before the matting net |
| Fresh alpha | network output | **last key frame's alpha warped by the flow** (`Warp` op) | network output |
| Warp previous stabilised alpha + flow-gated stabiliser blend (`Stabilize` op), re-upscale | ✓ | ✓ | ✓ |
| Display input + composite | ✓ | ✓ | ✓ |

(The first frame after attaching only seeds the stabiliser state,
`renderer.ts:409-413`.) On WebGPU the renderer also samples GPU time on every
5th network run (`backend.sync()`, `renderer.ts:367`), as it does in production.

**Verified by instrumentation**: the harness wraps the `run()` of the
renderer's own `TierModel` and `OpticalFlowNet` objects and counts calls in
every timed 192-frame pass (table in the results file). Every run of every
preset counted: xs 64 network + 128 flow runs; small and medium 96 + 96; large
and xl 192 + 192; identical on WebGPU and WebGL2, at both resolutions.

So the per-frame cost of the shipped pipeline is not "network ÷ (skipFrames+1)":
the off frames run the flow network and the warps, and large/xl run the flow
network **in addition to** the full network on every frame.

### Output path (why the first 2026-11-15 run capped at ~100–150 fps)

In the first run, passthrough reached 1,155 fps at 360p, but the canvas-copy
pipelines (no model) only 99–154 fps: getting a canvas-rendered frame into the
encoder cost more than any model. `web/ceiling.html` / `tools/ceiling.mjs`
isolate it (no model in any route; raw output in
`results/2026-11-15-apple-m4-v2-output-path.json`). Same M4, Chrome
157.0.8103.51 headless, 360p unless noted, median of 3:

| Route to the H.264 encoder (no model) | FPS |
|---|---:|
| Decoded frame (passthrough) | 1,058 |
| `new VideoFrame(decodedFrame)` | 1,054 |
| `VideoFrame` over a CPU RGBA buffer / NV12 buffer | 1,103 / 1,056 |
| WebGPU → `OffscreenCanvas` → `new VideoFrame(canvas)` (the first run's path) | 96 |
| WebGPU → on-page `<canvas>` → `VideoFrame` | 100 |
| WebGPU → `transferToImageBitmap()` → `VideoFrame` | 98 |
| WebGL2 → `OffscreenCanvas` → `VideoFrame` | 90 |
| 2D `drawImage` → `OffscreenCanvas` → `VideoFrame` | 84 |
| WebGPU canvas, encoder `prefer-hardware` / `prefer-software` / `latencyMode: 'realtime'` | 95 / 98 / 98 |
| Encoder alone (own loop, no decoder), WebGPU canvas frames: default / hw / sw / realtime | 78 / 76 / 75 / 80 |
| Encoder alone, CPU RGBA frames: default / sw | 1,081 / 2,904 |
| `VideoFrame(canvas).copyTo(RGBA)` alone, 360p / 720p | 790 / 295 |
| **Readback stage** (`copyTo` → RGBA `VideoFrame` → encoder), 360p: passthrough / WebGPU / WebGL2 / 2D canvas | 708 / 510 / 437 / 635 |
| **Readback stage**, 720p: passthrough / WebGPU / WebGL2 / 2D canvas | 267 / 230 / 196 / 251 |

Findings:

- **The cost is the encoder consuming a canvas-backed `VideoFrame`**, ~10–13 ms
  per frame whatever the canvas type (WebGPU / WebGL2 / 2D, `OffscreenCanvas`
  or on-page `<canvas>`, or via `transferToImageBitmap`), the resolution, or the
  encoder settings (`hardwareAcceleration` prefer-hardware / prefer-software,
  `latencyMode: 'realtime'`). Even fed directly by a `VideoEncoder` with
  event-driven backpressure (no stream helpers, no `setTimeout` polling), canvas
  frames encode at 73–81 fps, while CPU-memory RGBA or NV12 frames encode at
  >1,000 fps (hardware) and ~3,000 fps (software). Creating the `VideoFrame`
  from the canvas costs ~0.2 ms; the time goes inside the encoder.
- **Reading the same frames back is fast**: `VideoFrame.copyTo()` to an RGBA
  buffer runs at ~660–840 fps (360p) and ~300 fps (720p), even one copy at a
  time (1, 2, 4 or 8 in flight made no difference).
- `latencyMode: 'realtime'` lowers passthrough (624 vs 1,058 fps) and doesn't
  help canvas frames; `prefer-software` encodes CPU frames ~3× faster but canvas
  frames no faster.
- **Headless vs headed** was not re-tested (it would open a window on a machine
  with a screen-recording app open); the 2026-11-15 run found them within ~5%.
- **Chrome version**: the webrtcHacks harness, same output path, got 358 fps
  with Chrome 156 in October and 133 fps with 157 on 2026-11-15. A Chrome 156
  build was not run side by side this time, so a Chrome 157 regression vs a
  change in the machine's state (see Conditions: a screen recorder with the
  camera open, WindowServer using ~30% of the GPU) is **not resolved**.

**What the harness does now**: an output stage between `process()` and the
encoder, the same for every pipeline including passthrough
(`readbackStream` in `web/src/bench.js`): `copyTo()` each frame's pixels into
CPU memory (async, up to 4 frames in flight, order kept), and give the encoder
a new RGBA `VideoFrame` over that buffer. In the v2 sessions it raised the
no-model ceiling from ~95 to 551–586 fps (WebGPU canvas copy, 360p) and 224–228
fps (720p), at a cost every pipeline pays equally (passthrough drops from
~1,050 to 739–789 fps at 360p, because its NV12 frames are now converted and
copied too). WebGL2 canvases read back more slowly than WebGPU ones (427–451 vs
551–586 fps at 360p), so the WebGL2 canvas-copy row is the ceiling for
WebGL-composited pipelines (row 1 GPU delegate; rows 3–5 on WebGL2). `npm run bench -- --output direct` restores the old path.

This is a workaround a real app could also use, not a special path for any
pipeline. It does not make transcode FPS a measure of GPU time alone: the
readback waits for each frame's GPU work, and its cost grows with resolution.

### What's timed

**Primary: transcode FPS**: cached `EncodedVideoChunk`s → `VideoDecoder` →
`process()` → readback output stage → `VideoEncoder` (H.264, `webcodecs-utils`
`getCodecString` / `getBitrate`, browser-chosen hardware acceleration) → MP4
muxer, as fast as backpressure allows. FPS = 192 frames / wall clock from the
first chunk to the finalised file. The encoder consumes real pixels whatever
the backend, so it is the one arbiter of throughput that is comparable across
WebGPU, WebGL2 and WASM, whereas each backend's "wait for the GPU" (GL fence,
`onSubmittedWorkDone`, synchronous WASM) is a different mechanism.

**Diagnostics only** (not comparable across backends for that reason):
`process` = decode → `process()` → wait for the output's pixels (draw it into a
1×1 canvas, read the pixel), one frame at a time, no encoder; `model` = model
work only per frame, GPU-synchronised (Longpipe: `Input` op + network; MediaPipe
Tasks: `segmentForVideo()`, which includes its resize and mask upsampling;
LiteRT: resize pass + `model.run`). Row 5 has no model-only mode (its per-frame
work isn't a single model call).

**Protocol**: for each clip and mode, 5 rounds; each round measures every
pipeline once, in an order rotated by one position per round, on a freshly
loaded page: create the pipeline → one untimed warm-up pass over the clip →
one timed pass → destroy. Interleaving spreads drift (thermals, background
load) over all pipelines instead of the ones that ran last; recreating the
pipeline each round keeps ≤ 1 WebGL context / model alive at a time. Median and
min–max of the 5 timed passes are reported. Not timed: demuxing, pipeline
creation.

**Background load guard**: the M4 is also a production worker for an unrelated
GPU service. `--busy-cmd` polls a command once a second (here: whether a worker
job process exists); any measurement whose timed pass overlapped a busy sample
(±1.5 s) is discarded and repeated after 5 s idle. An independent per-second
log of worker activity was kept during every session.

All assets (clip, MediaPipe models, Longpipe weights from the SDK's default
`cdn.longpipe.dev/models/v/0.0.5/`, the MediaPipe f16 packs from
`cdn.longpipe.dev/models/mediapipe/selfie/v1/`) are downloaded at `npm install`
and checked by sha256.

**Test clip**: `woman_on_video_call` (a person talking to a webcam), 1280×720
H.264, 192 frames at 24 fps, and a 640×360 re-encode. Recorded by Sam
Bhattacharyya and published with the webrtcHacks article harness; fetched from
that repo at a pinned commit.

**Correctness check**: `npm run bench -- --check` runs every pipeline on one
frame (the as-shipped rows see the 12 preceding frames first, so the checked
frame goes through the flow and stabiliser path) and reports the share of green
pixels and agreement with MediaPipe GPU. On the M4 every model pipeline replaced
64.9–66.2% of the frame and agreed with MediaPipe GPU on ≥98.9% of pixels;
outputs were inspected visually.

### Results (2026-11-15 v2, Apple M4, Chrome 157)

Full tables (min–max, diagnostics, old output path) and conditions:
[`performance/results/2026-11-15-apple-m4-v2.md`](performance/results/2026-11-15-apple-m4-v2.md).
All Longpipe rows f16 unless marked (the SDK default). An earlier first run was
capped by the old output path and has been removed.

Transcode FPS, median of 5 interleaved runs, **session 1 / session 2**, **360p · 720p**:

| Model | Engine | Pipeline | Mask | FPS (360p · 720p) |
|---|---|---|---|---:|
| – (no model) | – | passthrough (decode → encode) | – | 789 / 739 · 249 / 263 |
| – (no model) | WebGPU | canvas copy | – | 586 / 551 · 224 / 228 |
| – (no model) | WebGL2 | canvas copy | – | 451 / 427 · 185 / 195 |
| MediaPipe square | MediaPipe Tasks, GPU delegate | every frame | 256×256 | 118 / 114 · 93 / 94 |
| MediaPipe landscape | MediaPipe Tasks, GPU delegate | every frame | 256×144 | 123 / 120 · 93 / 97 |
| MediaPipe square | MediaPipe Tasks, CPU delegate | every frame | 256×256 | 117 / 114 · 69 / 70 |
| MediaPipe landscape | MediaPipe Tasks, CPU delegate | every frame | 256×144 | 150 / 153 · 81 / 82 |
| MediaPipe square | LiteRT.js, WebGPU | every frame | 256×256 | 380 / 381 · 201 / 196 |
| MediaPipe landscape | LiteRT.js, WebGPU | every frame | 256×144 | 414 / 411 · 196 / 204 |
| MediaPipe square | Longpipe, WebGPU | every frame | 256×256 | 206 / 192 · 132 / 130 |
| MediaPipe landscape | Longpipe, WebGPU | every frame | 256×144 | 209 / 205 · 130 / 132 |
| MediaPipe square | Longpipe, WebGL2 | every frame | 256×256 | 125 / 123 · 90 / 90 |
| MediaPipe landscape | Longpipe, WebGL2 | every frame | 256×144 | 122 / 123 · 88 / 92 |

| Longpipe model, Longpipe engine | xs (384×240) | small (384×224) | medium (512×320) | large (640×400) | xl (1280×768) |
|---|---:|---:|---:|---:|---:|
| **As shipped**, WebGPU | 308 / 288 · 159 / 161 | 291 / 298 · 155 / 162 | 241 / 237 · 140 / 146 | 150 / 157 · 104 / 108 | 91 / 88 · 69 / 69 |
| **As shipped**, WebGL2 | 205 / 192 · 122 / 129 | 195 / 189 · 121 / 125 | 154 / 159 · 104 / 105 | 102 / 101 · 76 / 78 | 63 / 60 · 49 / 52 |
| Network every frame, WebGPU | 273 / 280 · 154 / 157 | 261 / 277 · 145 / 156 | 217 / 216 · 133 / 141 | 203 / 199 · 129 / 132 | 119 / 121 · 89 / 93 |
| Network every frame, WebGL2 | 189 / 191 · 124 / 129 | 190 / 183 · 122 / 121 | 149 / 139 · 103 / 106 | 137 / 134 · 100 / 100 | 85 / 84 · 68 / 68 |
| Network every frame, WebGPU, f32 | | | | 162 / 162 · 107 / 110 | 82 / 82 · 64 / 66 |
| Network every frame, WebGL2, f32 | | | | 128 / 121 · 88 / 89 | 70 / 69 · 55 / 54 |

In short (orderings held in both sessions):

- **MediaPipe's model on LiteRT.js (WebGPU) is the fastest model pipeline
  tested**, faster than every Longpipe configuration (360p: 380–414 vs at most
  308 fps). On the same model, LiteRT.js is 1.8–2.0× faster than Longpipe's
  engine on WebGPU, which is ~1.7× faster than MediaPipe Tasks' GPU delegate.
- **Longpipe as shipped vs MediaPipe Tasks**: on WebGPU, xs/small/medium are
  faster than every Tasks configuration, large is level at 360p and faster at
  720p, **xl is slower** than Tasks' best configuration. On WebGL2, xs/small are
  faster, medium level, **large and xl slower** than Tasks' best.
- **The shipped flow schedule saves little**: xs/small/medium are only 0–14%
  faster than running the network on every frame, because off frames run the
  learned flow network, warps and stabiliser; large/xl are **18–27% slower**
  than the network alone, because they run the flow network on every frame as
  well. What the flow path buys (temporal stability) is not measured.
- At 720p the readback output path (~225 fps with no model) compresses the
  differences between the fastest pipelines.

### Caveats specific to performance

- **One machine, one browser version.** Results differ a lot across GPUs and
  browsers (the article harness found the WebGPU/WebGL ranking flips on Intel
  Iris Xe). Run it on yours.
- **The output path is part of the number.** Transcode FPS uses the readback
  stage described above for every pipeline; on this machine and Chrome version
  the plain canvas → encoder path caps everything at ~75–110 fps. The readback
  alone runs at ~790 fps at 360p (~1.3 ms per frame) and ~300 fps at 720p
  (~3.3 ms), so at 720p it is close to the throughput of the fastest pipelines.
- **Not a quiet machine.** The M4 was in use: a production GPU worker (its
  jobs were detected, and overlapping measurements discarded and repeated), a
  screen recorder with the camera open, WindowServer at ~30% GPU, load average
  3.4–5.2. Absolute numbers on an idle machine would likely be higher; the
  model-only times of the same pipelines were 18–43% higher than in the first
  run the same afternoon. The two v2 sessions agreed within 7.5% per pipeline.
- **Main thread, no transports.** Every pipeline, including the as-shipped
  rows, runs on the page's main thread. The SDK runs its renderer in a worker
  fed by `MediaStreamTrack`s; that thread hop and transport cost are not
  measured, for any row.
- **Different pre/post-processing per engine.** MediaPipe Tasks does its own
  resize and returns the mask at frame size; LiteRT and the Longpipe engine get
  our/Longpipe's resize and composite. Each row is what that engine needs to
  produce the same effect.
- **Headless Chrome** (driven over SSH). On this Mac it reaches the real GPU
  (Metal adapter in the report); the first run found headed within ~5%; not
  re-tested this time.
- **Interleaved protocol.** Each measurement creates the pipeline, warms it up
  with one pass over the clip and times one pass. Steady-state over minutes
  (thermal throttling) is not measured.

---

## Caveats

- **Segmentation vs matting.** MediaPipe selfie is a segmentation model: it is
  trained to output a person/background probability, not a fractional alpha for
  hair and edges. P3M scores fractional alpha. Whole-image MAD therefore
  rewards a matting model at edges by design; IoU@0.5 (a segmentation metric) is
  reported so the comparison doesn't rest only on matting metrics, and it is
  where MediaPipe does best.
- **Resolution.** MediaPipe's masks are 256×256 or 256×144; Longpipe's are
  384×224 to 1280×768. Part of Longpipe's advantage is simply resolution: compare
  xs/small (closest to MediaPipe's size) with MediaPipe, not xl.
- **In-domain advantage, and checkpoint selection.** Longpipe was trained on the
  P3M-10k training split (among other data), whose images are face-blurred like
  P3M-500-P; MediaPipe was not. And Longpipe's training code selects the
  checkpoint with the lowest MAD on a validation set that **includes
  P3M-500-P** (plus a synthetic set). P3M-500-P is therefore not a clean held-out
  test set for Longpipe; **P3M-500-NP (not blurred, not used for checkpoint
  selection) is the cleaner comparison**, and both are reported separately.
  Longpipe's lead over MediaPipe is smaller on NP than on P.
- **Privacy-blurred faces in P3M-500-P.** Faces are blurred in P3M-500-P (and
  in the training split). A face detector-dependent or face-sensitive model may
  behave differently on them than on real webcam faces. P3M-500-NP has real faces.
- **Stills, not video.** P3M is still photographs, mostly not webcam-like (full
  body, outdoor, varied framing). Webcam framing and temporal stability (which
  the SDK's flow path and stabiliser address) are not measured.
- **ONNX fp32 vs browser.** Quality is measured with fp32 ONNX Runtime, one
  still image at a time. In the browser, `EffectsPipeline` runs every preset in
  f16 where the GPU supports it, and only xs/small/medium run the full network
  on every frame: on the other frames it propagates the last key frame's alpha
  with a learned optical-flow network, and every frame goes through a
  flow-gated temporal stabiliser (see [Performance](#longpipe-as-shipped-what-runs-on-each-frame)).
  None of that is scored here: P3M is single images, so it cannot measure the
  quality of flow-propagated frames or temporal stability, and no temporal
  metric is reported. Browser output was not scored against P3M.
- **The MediaPipe rows of the performance benchmark share one quality score.**
  Rows 1–3 there run the same MediaPipe selfie model on three engines (MediaPipe
  Tasks, LiteRT.js, Longpipe's); its P3M scores are the MediaPipe rows above.
  Longpipe's engine reproduces the TFLite interpreter's mask to a max abs error
  of 1.4e-5 in f32 (WebGPU and WebGL2, one real frame; commit 3882e0a, longpipe
  0.0.17); in f16, the SDK default used in the performance rows, the mean
  difference was ~5e-5. The f16 output was not scored on P3M.
- **Same compositing, different ownership.** The performance comparison holds
  the effect constant, but each library does its own pre/post-processing; see
  the pipeline table.

## Reproducing

See [`README.md`](README.md) for the exact commands. Report results with the
generated JSON unmodified; it records versions, hardware, and every run.
