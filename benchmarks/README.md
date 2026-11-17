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

## Results

The current headline numbers (Longpipe 0.0.18: speed charts, the P3M quality table and
what each row means) are in [`analysis/README.md`](../analysis/README.md), which the
main README and longpipe.dev/docs/benchmarks summarise. Full result files:

- Quality: [`quality/results/2026-11-15-p3m.md`](quality/results/2026-11-15-p3m.md)
  (all metrics, per-orientation breakdowns) and its per-image CSV.
- Performance (0.0.17, model × engine × pipeline, 360p/720p, min–max):
  [`performance/results/2026-11-15-apple-m4-v2.md`](performance/results/2026-11-15-apple-m4-v2.md);
  0.0.18 command-buffer A/B: [`performance/results/2026-11-15-apple-m4-engine-perf-ab-transcode.json`](performance/results/2026-11-15-apple-m4-engine-perf-ab-transcode.json).

Read [METHODOLOGY.md](METHODOLOGY.md) before quoting any number.

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
