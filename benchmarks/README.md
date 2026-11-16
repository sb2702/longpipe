# Benchmarks: Longpipe vs MediaPipe

Reproducible quality and speed comparisons between Longpipe and Google's MediaPipe
selfie segmenter. Everything here is meant to be re-run by anyone: the scripts
download pinned, hash-checked models, and the results files record versions,
hardware, and every individual measurement.

- [`quality/`](quality/): alpha-matte accuracy on the P3M-10k validation sets
  (Python, CPU).
- [`performance/`](performance/): speed in the browser, decode → background
  replacement → H.264 encode, organised as model × engine × pipeline: MediaPipe's
  selfie model on MediaPipe Tasks, LiteRT.js and Longpipe's engine; Longpipe's
  model with the network on every frame and as the SDK actually ships it.
- [`METHODOLOGY.md`](METHODOLOGY.md): what is measured, fairness caveats, and how
  these numbers relate to the charts previously shown in the README. **Read it
  before quoting any number.**

## Headline results

### Quality: P3M-10k validation, 2026-11-15

Mean absolute alpha error (MAD, lower is better) and IoU of the mask at 0.5
(higher is better), at ground-truth resolution, all 1,000 images (portraits
included). Full metrics (SAD, MSE, Grad, Conn, trimap-unknown band) and
per-orientation breakdowns: [`quality/results/2026-11-15-p3m.md`](quality/results/2026-11-15-p3m.md).

| Method | Mask res. | P3M-500-P MAD | P3M-500-P IoU | P3M-500-NP MAD | P3M-500-NP IoU |
|---|---|---:|---:|---:|---:|
| Longpipe xs | 384×240 | 0.0210 | 0.9229 | 0.0233 | 0.9262 |
| Longpipe small | 384×224 | 0.0170 | 0.9365 | 0.0155 | 0.9536 |
| Longpipe medium | 512×320 | 0.0106 | 0.9657 | 0.0102 | 0.9716 |
| Longpipe large | 640×400 | 0.0111 | 0.9632 | 0.0100 | 0.9720 |
| Longpipe xl | 1280×768 | **0.0071** | **0.9770** | **0.0080** | **0.9787** |
| MediaPipe selfie (square) | 256×256 | 0.0304 | 0.9228 | 0.0249 | 0.9398 |
| MediaPipe selfie (landscape) | 256×144 | 0.0322 | 0.9173 | 0.0276 | 0.9340 |

Every Longpipe preset has lower MAD than MediaPipe on both sets. On IoU,
MediaPipe (square) beats Longpipe **xs** on P3M-500-NP and ties it on P3M-500-P;
small and up beat it. P3M-500-NP is the cleaner comparison: Longpipe trained on
P3M-10k's (face-blurred) training split and selects checkpoints on a validation
set that includes P3M-500-P. See [METHODOLOGY.md](METHODOLOGY.md#caveats).

### Performance: Apple M4 MacBook Pro, Chrome 157 (v2)

Background replacement on a 192-frame webcam clip, organised as **model ×
engine × pipeline**: MediaPipe's selfie model on three engines (MediaPipe Tasks,
LiteRT.js, Longpipe's), and Longpipe's model with the network on every frame and
**as shipped** (the SDK's own renderer: full network on key frames, a learned
optical-flow network on the frames in between, a temporal stabiliser on every
frame). Headline metric: **transcode FPS**, WebCodecs decode → effect → H.264
encode → MP4, the same output path for every pipeline. Full results and
conditions: [`performance/results/2026-11-15-apple-m4-v2.md`](performance/results/2026-11-15-apple-m4-v2.md).

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

On this machine:

- **MediaPipe's model on LiteRT.js (WebGPU) is the fastest pipeline tested**,
  faster than every Longpipe preset. On the same model, LiteRT.js is 1.8–2.0×
  faster than Longpipe's engine on WebGPU, which is ~1.7× faster than MediaPipe
  Tasks.
- **Longpipe as shipped vs MediaPipe Tasks**: on WebGPU, xs–medium are faster
  than every Tasks configuration, large is level (360p) or faster (720p), xl is
  slower than Tasks' best configuration. On WebGL2, xs/small are faster, medium
  level, large and xl slower than Tasks' best.
- **The shipped flow schedule barely pays for itself in speed**: xs–medium are
  0–14% faster than the network on every frame; large/xl are 18–27% slower
  (they run the flow network on every frame too). Its purpose, temporal
  stability, is not measured here.
- One machine, in use while measured (a production GPU worker, whose jobs were
  detected and excluded, and other apps). Read [METHODOLOGY.md](METHODOLOGY.md#performance)
  for what each row runs, the output-path investigation, and caveats. The first
  run ([`2026-11-15-apple-m4.md`](performance/results/2026-11-15-apple-m4.md)) is
  kept as published; its transcode numbers were capped by the old output path.

## Quick start

### Quality

Requirements: Python 3.10+ (results were produced with 3.12), ~6 GB of disk for
the dataset, no GPU.

```sh
cd benchmarks/quality
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy pillow scipy onnxruntime mediapipe huggingface_hub

python eval_p3m.py --p3m-root /path/to/P3M-10k --limit 10    # quick check: 10 images per set
python eval_p3m.py --p3m-root /path/to/P3M-10k               # full run (~8 min on 8 CPU cores)
python eval_p3m.py --p3m-root /path/to/P3M-10k --protocol legacy-readme   # the old README chart's protocol
```

Options: `--methods longpipe-xl mediapipe-square …`, `--sets P3M-500-NP`,
`--workers N` (default: min(8, CPU count)), `--hf-revision` (default: the pinned
revision the published results used), `--out-dir`, `--tag`.

Writes `results/<date>-p3m.{md,json}` and a per-image `…-per-image.csv.gz`.
Models are cached in `quality/.cache/` (Longpipe ONNX from Hugging Face, ~150 MB
for all five presets; MediaPipe `.tflite` from Google, checked by sha256).

### Performance

Requirements: Node 20+, Google Chrome (or set `CHROME=/path/to/chrome`), a GPU.

```sh
cd benchmarks/performance
npm install                    # also downloads the test clip, MediaPipe models and Longpipe weights (sha256-checked)

npm run bench -- --check       # sanity-check every pipeline's output first
npm run bench                  # 360p + 720p transcode, every pipeline, 5 interleaved rounds
npm run bench -- --modes process,model --runs 3          # diagnostics
npm run bench -- --samples 360p --pipelines passthrough,shipped-webgpu-small,litert-webgpu,mediapipe-gpu
node tools/report.mjs s1=results/<file>.json [s2=results/<file2>.json]   # markdown tables
node tools/ceiling.mjs --headless  # output-path investigation (see METHODOLOGY.md)
node tools/profile.mjs configs.json out.json   # per-dispatch GPU profile of one network (see below)
```

`tools/profile.mjs` (page `web/profile.html`, code `web/src/profile.js`) builds
one network from this repo's `src/` (or LiteRT.js) and reports latency,
back-to-back throughput, CPU time per run, queue submits / passes / dispatches
per run, and each dispatch's GPU time from timestamp queries (headless Chrome
with `--enable-webgpu-developer-features`). `configs.json` is a list such as
`[{"impl":"longpipe","backend":"webgpu","net":"mp-square","dtype":"f16"},
{"impl":"litert","net":"mp-square"}]`; `net` is `mp-square`, `mp-landscape` or
a tier; `"frame": {"w": 640, "h": 360}` profiles the whole per-frame path
(Input → network → RenderOp) instead; `"exclude": {"name": "<label regex>"}`
also reports the one-pass GPU time without those dispatches (what fusing them
away could save at most).

Add `--headless` to run without a window (fine on macOS; on Linux see
METHODOLOGY.md). `--output direct` hands canvas frames straight to the encoder
instead of the readback stage (see METHODOLOGY.md, "Output path").
`--busy-cmd '<command>'` discards and repeats any measurement that overlapped a
moment when the command exited 0 (e.g. a background GPU job is running). Or run
`npm run dev` and open http://localhost:5174 to pick pipelines and run them in
your own browser, then **Download JSON**.

Pipeline names: `passthrough`, `canvas-copy-{webgpu,webgl}` (ceilings);
`mediapipe-{gpu,cpu}[-landscape]` (MediaPipe Tasks); `litert-webgpu[-landscape]`
(LiteRT.js); `lp-mediapipe-{webgpu,webgl}-{square,landscape}` (MediaPipe model on
Longpipe's engine); `longpipe-{webgpu,webgl}-{xs,small,medium,large,xl}[-f32]`
(network every frame); `shipped-{webgpu,webgl}-{xs,…,xl}` (the SDK's own
renderer, as shipped).

Close other apps, plug in power, and check the passthrough and canvas-copy rows:
they show how much of the frame budget the decoder/output path/encoder leave on
your machine.

## Getting P3M-10k

P3M-10k is published by its authors (Li et al., University of Sydney) for
research use, under a dataset release agreement (the agreement states an MIT
license, requires citing the P3M paper in work that uses the dataset, and notes
that the image copyrights stay with their original owners). It is **not** included
in this repository and must not be committed to it, not even as example images.

1. Read the release agreement:
   [P3M-10k_Dataset_Release_Agreement.pdf](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf).
   The authors ask that you only download the dataset if you agree to it.
2. Download **`P3M-10k.zip`** (~5.4 GB) from the link in the
   ["PPT Setting and P3M-10k Dataset"](https://github.com/JizhiziLi/P3M#ppt-setting-and-p3m-10k-dataset)
   section of [github.com/JizhiziLi/P3M](https://github.com/JizhiziLi/P3M):
   Google Drive, or Baidu Wangpan (password given there).
3. Unzip it anywhere. Use the extracted `P3M-10k` folder, unchanged, as
   `--p3m-root`:

```
P3M-10k/                       ← pass this folder as --p3m-root
├── train/                     (not used by the benchmark)
│   ├── blurred_image/
│   └── mask/
├── validation/
│   ├── P3M-500-P/
│   │   ├── blurred_image/     500 × .jpg   (face-blurred)
│   │   ├── mask/              500 × .png   (alpha matte)
│   │   └── trimap/            500 × .png
│   └── P3M-500-NP/
│       ├── original_image/    500 × .jpg   (not blurred)
│       ├── mask/              500 × .png
│       └── trimap/            500 × .png
├── train_list.txt
├── P3M-500-P_list.txt
├── P3M-500-NP_list.txt
├── P3M-10k_Release_Agreement.pdf
└── readme.txt
```

Only `validation/` is read. The script stops with an explanation if a folder is
missing, and prints the number of images it found in each set (500 + 500, of
which 499 are landscape) before evaluating.

Please cite the dataset if you publish results from it:

> Jizhizi Li, Sihan Ma, Jing Zhang, and Dacheng Tao. "Privacy-Preserving Portrait
> Matting." ACM MM 2021.

## Credits

The performance harness is adapted from the
[webrtcHacks WebGPU segmentation benchmark](https://github.com/sb2702/webrtc-hacks-webgpu/tree/main/segmentation)
(MIT), including its MediaPipe pipelines, frame I/O shaders, timing code and test
clip. MediaPipe models © Google, Apache 2.0.
