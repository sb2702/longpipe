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

Full tables: [`quality/results/2026-11-15-p3m.md`](quality/results/2026-11-15-p3m.md)
(standard) and [`quality/results/2026-11-15-p3m-legacy.md`](quality/results/2026-11-15-p3m-legacy.md).

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

An earlier version of the README showed a bar chart, "Mean Absolute Error,
Background Segmentation (P3M 10k Validation set)": XL 0.0077, L 0.0108, M 0.0109,
S 0.0125, XS 0.0139, MediaPipe Selfie 0.0307, described as the "average alpha
error on the P3M-10K validation set (499 images)… every Longpipe variant beats
MediaPipe on both MAD and IoU". The current README's performance/accuracy scatter
plot uses the same accuracy figures.

That chart came from `training/benchmark_accuracy.py` in the (private) training
repo, run in May 2026 (results file `benchmark_accuracy.json`, removed from that
repo in a later reorganisation). Its protocol:

- **499 images = the landscape images of P3M-500-P + P3M-500-NP** (246 + 253 with
  w/h > 1.05), not a 500-image set with one image missing. Portrait and square
  images were excluded "since `selfie_segmenter_landscape` is the model an end
  user actually picks for webcam input".
- Each image (and its mask) **centre-cropped to 16:9**, then resized to the model
  input; scored at the cropped resolution.
- **MediaPipe landscape model only.** MAD and IoU@0.5.
- **The Longpipe models of May 2026**, as PyTorch checkpoints, at that era's preset
  definitions: xl at 512×288, large/medium/compact/small at 256×144 (medium =
  large's checkpoint in fp16), xs at 192×108, plus xxs and compact presets that no
  longer exist. The chart's "M 0.0109" matches the `compact` row (medium was
  0.0108). The shipped models have been retrained and re-architected since
  (U-Net wrapper, new canvas sizes: see `TIER_CONFIG`).

Re-running that protocol today (`--protocol legacy-readme`) with the shipped ONNX
models:

| | old chart (May 2026 models) | legacy protocol, shipped models | standard protocol, all 1,000 images |
|---|---|---|---|
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

### What runs

Every pipeline has the same interface, `process(VideoFrame) → VideoFrame`, and
does the same effect: replace the background with solid green,
`mix(green, frame, alpha)`, with the alpha bilinearly upsampled to the frame size,
on the GPU, output as a new `VideoFrame` at the input resolution.

| Pipeline | Model runs on | Pre-processing | Composite |
|---|---|---|---|
| Longpipe {xs…xl} — WebGPU | `longpipe/model` `WebGPUBackend` (the npm package, v0.0.16) | SDK `Input` op (frame → canvasRes, zero-copy `importExternalTexture`) | SDK `RenderOp`, `{ mode: 'solid' }`, bilinear upscaler |
| Longpipe {xs…xl} — WebGL2 | `longpipe/model` `WebGLBackend` | SDK `Input` op | SDK `RenderOp` |
| MediaPipe — GPU (WebGL) | `@mediapipe/tasks-vision` 1.1.1 `ImageSegmenter`, GPU delegate | MediaPipe's own | our WebGL shader, in MediaPipe's context, on its mask texture (no readback) |
| MediaPipe — CPU (WASM) | same, CPU delegate (XNNPACK) | MediaPipe's own | mask (`Float32Array`) uploaded, our WebGPU shader |
| Passthrough | none | | | 
| Canvas copy — WebGPU / WebGL2 | none: alpha = 1 | | same composite shaders as the MediaPipe pipelines |

MediaPipe runs with both the square and the landscape selfie model (same files and
hashes as the quality benchmark).

- **Longpipe uses the model layer, not `EffectsPipeline`.** No worker, no autotune,
  no temporal stabiliser, no frame skipping: the network runs on **every** frame
  at the preset's resolution. (The production pipeline runs xs every 3rd frame and
  small/medium every 2nd, and adds the optical-flow stabiliser.) This is the
  apples-to-apples model cost, not what `EffectsPipeline` costs end to end.
- Precision per preset is the SDK's production choice (`src/pipeline/presets.ts`):
  xs, small, medium in **f16**; large, xl in **f32**. `--dtype f16|f32` overrides.
  MediaPipe's GPU precision isn't exposed by its API.
- The network input is stretched to the tier's canvas (8:5 or 12:7 for a 16:9
  clip), as MediaPipe stretches to 256×256 / 256×144.
- MediaPipe's pre/post-processing can't be swapped out, and it returns the mask
  resampled to the frame's resolution; that cost is part of using MediaPipe, so
  it's counted. Longpipe's `RenderOp` also uploads the frame into a tensor at
  frame resolution before compositing; also counted.
- The MediaPipe pipelines and the frame I/O shaders are taken from Sam
  Bhattacharyya's [webrtcHacks article harness](https://github.com/sb2702/webrtc-hacks-webgpu/tree/main/segmentation)
  (MIT), with the same encoder settings and timing code.

### What's timed

Three modes, each on a 360p and a 720p clip:

1. **`transcode` (end-to-end FPS)**: cached `EncodedVideoChunk`s →
   `VideoDecoder` → `process()` → `VideoEncoder` (H.264, `webcodecs-utils`
   `getCodecString` / `getBitrate`, browser-chosen hardware acceleration) → MP4
   muxer, as fast as backpressure allows. FPS = frames / wall-clock from the first
   chunk to the finalised file. One untimed warm-up pass, then N timed passes
   (default 5); **median** reported, min/max and every run in the JSON.
2. **`process` (effect latency)**: decode → `process()` → wait for the output
   frame's pixels (draw it into a 1×1 canvas and read the pixel), one frame at a
   time, no encoder. Reported as FPS (= 1000 / mean ms) and median ms per frame.
3. **`model` (model-only latency)**: on 48 decoded frames, cycled: submit the
   model work only and wait for the GPU to finish it, one frame at a time; 30
   warm-up iterations, then N runs of 200; median ms. Longpipe: `Input` op + network
   (alpha left at the tier's canvas resolution; no upscale, no composite);
   completion via `device.queue.onSubmittedWorkDone()` (WebGPU) or a GL fence
   (WebGL2). MediaPipe: `segmentForVideo()` (includes MediaPipe's resize and its
   upsampling of the mask to frame size, which its API doesn't separate);
   completion via a GL fence (GPU delegate) or on return (CPU delegate).

Not timed: demuxing, pipeline creation (weights are loaded from the local dev
server; shader compilation; reported as `initMs`).

All assets (clip, MediaPipe models, Longpipe weights from
`cdn.longpipe.dev/models/v/0.0.5/`, the SDK's default) are downloaded at
`npm install` and checked by sha256, so nothing is fetched mid-benchmark.

**Test clip**: `woman_on_video_call` (a person talking to a webcam), 1280×720
H.264, 192 frames at 24 fps, and a 640×360 re-encode. Recorded by Sam
Bhattacharyya and published with the webrtcHacks article harness; fetched from
that repo at a pinned commit. No CC-licensed clip of a person was needed or used.

**Correctness check**: `npm run bench -- --check` runs every pipeline on one
frame and reports the share of green (background) pixels and agreement with
MediaPipe GPU; on the M4 every pipeline replaced 65–66% of the frame and agreed
with MediaPipe on ≥99% of pixels, and the outputs were inspected visually.

### Results (2026-11-15, Apple M4, Chrome 157)

Full tables and observations:
[`performance/results/2026-11-15-apple-m4.md`](performance/results/2026-11-15-apple-m4.md).
In short, orderings that held in both sessions:

- Longpipe **WebGPU xs, small, medium, large** ran faster than every MediaPipe
  configuration (per-frame process mode; transcode at or above MediaPipe).
  large vs MediaPipe CPU + landscape model was close (185 vs 180 fps in session 2).
- Longpipe **xl** (1280×768) was slower than every MediaPipe configuration.
- On **WebGL2**, only Longpipe xs and small were clearly faster than MediaPipe;
  medium was level with MediaPipe's fastest configuration, large slower, xl the
  slowest pipeline tested.
- MediaPipe's fastest configuration was the CPU (WASM) delegate with the
  landscape model, not its GPU delegate.

### Caveats specific to performance

- **One machine, one browser version.** Results differ a lot across GPUs and
  browsers (the article harness found the WebGPU/WebGL ranking flips on Intel
  Iris Xe). Run it on yours.
- **The output path dominates transcode FPS on the M4 tested.** With Chrome 157,
  getting a canvas-rendered frame into the H.264 encoder cost ~6–10 ms per frame
  whatever the resolution: the "canvas copy" pipelines (no model at all) reached
  only 97–159 fps, against 400–1,150 fps for passthrough. That compresses every
  effect pipeline's transcode FPS into a narrow band. The webrtcHacks harness run
  on the same machine the same day showed the same effect (its own WebGPU
  pipeline: 133 fps now vs 358 fps published with Chrome 156). Hence process- and
  model-mode numbers are also reported; they show the differences that transcode
  FPS hides.
- **Session-to-session drift.** On the same machine, the same pipelines moved by
  up to ~50% (transcode) and ~25% (process, uniformly across pipelines) between
  two sessions 20 minutes apart, while runs within a session agreed to a few
  percent. Two sessions are published; compare pipelines within a session, and
  trust orderings that hold in both, not absolute FPS.
- **Background load.** The MacBook had other applications open, and a
  background GPU job service that was idle except for ~5 s during session 1
  (its activity was logged every second; the affected measurements were re-run).
  See the results file.
- **Headless Chrome** was used (driven over SSH). On this Mac it reaches the real
  GPU (Metal adapter in the report), and a headed comparison gave the same numbers
  within ~5%. On Linux, headless WebGPU needs extra flags (see the article
  harness's methodology).
- Sequential order: each pipeline is created, warmed up, timed, destroyed in turn.

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
  the SDK's flow stabiliser addresses) are not measured.
- **ONNX fp32 vs browser.** Quality is measured with fp32 ONNX Runtime; in the
  browser xs/small/medium run in f16 and the pipeline adds frame skipping and
  temporal stabilisation. Browser output was not scored against P3M.
- **Same compositing, different ownership.** The performance comparison holds
  the effect constant, but each library does its own pre/post-processing; see
  the pipeline table.

## Reproducing

See [`README.md`](README.md) for the exact commands. Report results with the
generated JSON unmodified; it records versions, hardware, and every run.
