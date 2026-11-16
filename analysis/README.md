# Longpipe vs MediaPipe selfie segmentation

How Longpipe compares with Google's MediaPipe selfie segmenter, the default most web apps use for
virtual backgrounds, on speed and on matte quality. Every number here comes from code in
[`benchmarks/`](../benchmarks/) that anyone can re-run. Read
[`benchmarks/METHODOLOGY.md`](../benchmarks/METHODOLOGY.md) for the full details and caveats.

Three things decide how fast and how good a background effect is:

- **Model:** the neural network and its weights (MediaPipe selfie, or Longpipe's xs → xl presets).
- **Engine:** what runs the network in the browser. For MediaPipe that's its own
  `@mediapipe/tasks-vision` library; Longpipe has its own hand-written WebGPU/WebGL shaders.
- **Pipeline:** what happens around the network. Longpipe runs its full network on key frames
  and a small learned optical-flow network on the frames in between, plus a temporal stabiliser.

The tables compare what a developer gets out of the box: MediaPipe's tasks-vision library against
the Longpipe SDK as shipped. They also include MediaPipe's model running on Longpipe's engine
(`model: 'mediapipe-selfie'`, available since 0.0.17), to separate the engine from the model.

## Performance

Background replacement on a 192-frame webcam clip at **360p**, on an Apple M4 MacBook Pro with
Chrome 157. Frames per second for the full transcode (WebCodecs decode → effect → H.264 encode),
median of 5 interleaved runs. Higher is better.

| Model and engine | WebGPU | WebGL |
|---|---:|---:|
| MediaPipe selfie (landscape), tasks-vision GPU delegate | – | 123 |
| MediaPipe selfie (landscape), Longpipe engine | 209 | 122 |
| Longpipe xs | 308 | 205 |
| Longpipe small | 291 | 195 |
| Longpipe medium | 241 | 154 |
| Longpipe large | 150 | 102 |
| Longpipe xl | 91 | 63 |

MediaPipe's tasks-vision library has no WebGPU delegate. Its GPU delegate runs on WebGL.

The Longpipe rows run the SDK's own renderer, exactly as shipped:
- **xs:** the full network on every 3rd frame, the flow network on the other two.
- **small and medium:** the full network on every 2nd frame, the flow network on the others.
- **large and xl:** both networks on every frame.

The harness counts these runs to confirm the schedule. The 720p results, MediaPipe's CPU delegate,
LiteRT.js and the run-to-run spread are in
[`benchmarks/performance/results/2026-11-15-apple-m4-v2.md`](../benchmarks/performance/results/2026-11-15-apple-m4-v2.md).

### Reproduce

Node 20+, Google Chrome, a GPU:

```sh
cd benchmarks/performance
npm install                      # downloads the test clip, MediaPipe models and Longpipe weights (sha256-checked)
npm run bench -- --check         # sanity-check each pipeline's output first
npm run bench -- --samples 360p --pipelines mediapipe-gpu-landscape,lp-mediapipe-webgpu-landscape,lp-mediapipe-webgl-landscape,shipped-webgpu-xs,shipped-webgpu-small,shipped-webgpu-medium,shipped-webgpu-large,shipped-webgpu-xl,shipped-webgl-xs,shipped-webgl-small,shipped-webgl-medium,shipped-webgl-large,shipped-webgl-xl
```

Code:
- [`benchmarks/performance/web/src/pipelines.js`](../benchmarks/performance/web/src/pipelines.js):
  each pipeline, all with the same `process(VideoFrame) → VideoFrame` interface.
- [`benchmarks/performance/tools/bench.mjs`](../benchmarks/performance/tools/bench.mjs): the runner.

Every pipeline uses the same output stage, so the comparison measures only the effect. Your numbers
will differ on other hardware and browsers; the ordering is what to compare.

## Quality

Matte accuracy on the [P3M-10k](https://github.com/JizhiziLi/P3M) validation sets: 1,000 portrait
photos with hand-labelled alpha mattes. Each image is scored at full resolution with the P3M paper's
own metric code.

- **MAD:** mean absolute alpha error. Lower is better.
- **IoU:** overlap of the mask thresholded at 0.5. Higher is better.

| Model | Mask resolution | P3M-500-P MAD | P3M-500-NP MAD | P3M-500-P IoU | P3M-500-NP IoU |
|---|---|---:|---:|---:|---:|
| Longpipe xs | 384×240 | 0.0210 | 0.0233 | 0.923 | 0.926 |
| Longpipe small | 384×224 | 0.0170 | 0.0155 | 0.937 | 0.954 |
| Longpipe medium | 512×320 | 0.0106 | 0.0102 | 0.966 | 0.972 |
| Longpipe large | 640×400 | 0.0111 | 0.0100 | 0.963 | 0.972 |
| Longpipe xl | 1280×768 | **0.0071** | **0.0080** | **0.977** | **0.979** |
| MediaPipe selfie (square) | 256×256 | 0.0304 | 0.0249 | 0.923 | 0.940 |
| MediaPipe selfie (landscape) | 256×144 | 0.0322 | 0.0276 | 0.917 | 0.934 |

MediaPipe's model scores the same on either engine: Longpipe's implementation matches Google's
TFLite reference to within 1.4e-5 (f32).

- **Every Longpipe preset has a lower matte error than MediaPipe.**
- **On IoU,** MediaPipe's square model beats xs on P3M-500-NP and ties it on P3M-500-P. Small and
  up beat it.

Two notes on fairness:
- MediaPipe selfie is a segmentation model, scored here on a matting task. That's deliberate: a
  virtual background shows the matte, so matte quality is what users see.
- Longpipe was trained on P3M-10k's (face-blurred) training split, so **P3M-500-NP is the cleaner
  comparison.**

This test is single images, so it doesn't measure temporal stability (flicker), which is what
Longpipe's flow pipeline is for.

### Reproduce

Python 3.10+, no GPU. Download P3M-10k yourself, following the instructions in
[`benchmarks/README.md`](../benchmarks/README.md#getting-p3m-10k). The dataset is not in this repo.

```sh
cd benchmarks/quality
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python eval_p3m.py --p3m-root /path/to/P3M-10k --limit 10   # quick check
python eval_p3m.py --p3m-root /path/to/P3M-10k              # full run, ~8 min on 8 CPU cores
```

Code: [`benchmarks/quality/eval_p3m.py`](../benchmarks/quality/eval_p3m.py) runs Longpipe through
its official ONNX exports and MediaPipe through Google's `mediapipe` Python package, using the same
`.tflite` file as the web library. Full metrics (SAD, MSE, gradient, connectivity) are in
[`benchmarks/quality/results/2026-11-15-p3m.md`](../benchmarks/quality/results/2026-11-15-p3m.md).
