# Benchmarks: Longpipe vs MediaPipe

Reproducible quality and speed comparisons between Longpipe and Google's MediaPipe
selfie segmenter. Everything here is meant to be re-run by anyone: the scripts
download pinned, hash-checked models, and the results files record versions,
hardware, and every individual measurement.

- [`quality/`](quality/): alpha-matte accuracy on the P3M-10k validation sets
  (Python, CPU).
- [`performance/`](performance/): speed in the browser, decode → background
  replacement → H.264 encode, with the SDK's model layer and MediaPipe Tasks for
  Web.
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

### Performance: Apple M4 MacBook Pro, Chrome 157

Background replacement on a 192-frame webcam clip, median of 5 runs, two
sessions ("s1 / s2"). **Transcode** = WebCodecs decode → effect → H.264 encode
(end-to-end). **Process** = decode → effect → wait for the output frame, no
encoder. **Model ms** = model only, GPU-synchronised. Longpipe runs the network
on every frame via `longpipe/model`. Full results and conditions:
[`performance/results/2026-11-15-apple-m4.md`](performance/results/2026-11-15-apple-m4.md).

| Pipeline | Mask | 360p transcode FPS | 720p transcode FPS | 360p process FPS | 720p process FPS | Model ms (360p) |
|---|---|---:|---:|---:|---:|---:|
| Passthrough (decode → encode) | – | 1155 / 1082 | 424 / 399 | 1885 / 1446 | 706 | – |
| Canvas copy, WebGPU | – | 99 / 154 | 97 / 159 | 1034 / 769 | 401 | – |
| Longpipe xs — WebGPU (f16) | 384×240 | 101 / 147 | 121† / 108 | 449 / 332 | 179 | 1.38 |
| Longpipe small — WebGPU (f16) | 384×224 | 111 / 148 | 125† / 109 | 421 / 307 | 170 | 1.50 |
| Longpipe medium — WebGPU (f16) | 512×320 | 98 / 137 | 100 / 116 | 336 / 246 | 144 | 2.16 |
| Longpipe large — WebGPU (f32) | 640×400 | 123 / 147 | 123 / 103 | 236 / 185 | 107 | 3.35 |
| Longpipe xl — WebGPU (f32) | 1280×768 | 95 / 92 | 88 / 62 | 108 / 90 | 53 | 8.29 |
| Longpipe xs — WebGL2 (f16) | 384×240 | 77 / 110 | 74 / 106 | 284 / 194 | 121 | 2.81 |
| Longpipe small — WebGL2 (f16) | 384×224 | 75 / 111 | 75 / 104 | 275 / 227 | 108 | 2.84 |
| Longpipe medium — WebGL2 (f16) | 512×320 | 74 / 104 | 74 / 99 | 219 / 183 | 87 | 4.23 |
| Longpipe large — WebGL2 (f32) | 640×400 | 73 / 99 | 97 / 93 | 177 / 145 | 74 | 5.30 |
| Longpipe xl — WebGL2 (f32) | 1280×768 | 70 / 65 | 46 / 60 | 86 / 73 | 43 | 10.63 |
| MediaPipe square — GPU (WebGL) | 256×256 | 83 / 104 | 76 / 95 | 166 / 147 | 79 | 5.06 |
| MediaPipe landscape — GPU (WebGL) | 256×144 | 86 / 102 | 73 / 103 | 175 / 136 | 80 | 4.97 |
| MediaPipe square — CPU (WASM) | 256×256 | 67 / 89 | 83 / 83 | 151 / 133 | 75 | 4.92 |
| MediaPipe landscape — CPU (WASM) | 256×144 | 76 / 77 | 84 / 87 | 209 / 180 | 89 | 3.19 |

† overlapped a 5 s background GPU job in session 1; use the session 2 value.

On this machine the encoder path caps transcode FPS (the canvas-copy row has
no model at all) and absolute numbers drifted between sessions, but the
process-mode orderings held in both sessions: Longpipe WebGPU xs–large are faster
than every MediaPipe configuration (large only narrowly in session 2); **Longpipe
xl is slower than all of them**; and on **WebGL2 only xs and small are clearly
faster** than MediaPipe (medium is level, large slower). In end-to-end transcode
the gaps are much smaller.
One machine, one browser: run it on yours.

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
npm run bench                  # 360p + 720p; transcode, process and model modes; 5 runs each
npm run bench -- --samples 360p --modes transcode --pipelines passthrough,longpipe-webgpu-small,mediapipe-gpu
node tools/report.mjs results/<file>.json   # markdown tables from a results file
```

Add `--headless` to run without a window (fine on macOS; on Linux see
METHODOLOGY.md). `--dtype f16|f32` forces Longpipe's precision (default: the SDK
preset's). Or run `npm run dev` and open http://localhost:5174 to pick
pipelines and run it in your own browser, then **Download JSON**.

Close other apps, plug in power, and check the passthrough and canvas-copy rows:
they show how much of the frame budget the decoder/encoder leave on your
machine.

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
