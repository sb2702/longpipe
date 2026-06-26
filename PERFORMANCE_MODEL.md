# Performance Model & WebGPU Expectations

Can Longpipe match (or beat) the original Vectorly performance result with WebGPU, across
the effects we cataloged — both individually and composed in parallel? This note decomposes
where the gain actually comes from, maps it per effect class, and specifies what the
benchmark harness must measure to *prove* (not assert) it.

Reference: [Vectorly blog — ~3% vs ~12% CPU, zero-copy GPU pipeline](https://sambhattacharyya.com/blog/building-a-more-efficient-background-segmentation-model-than-google).
Companion to [ROADMAP_EXECUTION.md](ROADMAP_EXECUTION.md) (Phase 0.4 / 1.3 / 1.4) and
[EFFECTS_LANDSCAPE.md](EFFECTS_LANDSCAPE.md).

## 1. Decompose the original gain (it has two levers, they generalize differently)

The blog's headline is a **CPU** metric: **~3% CPU** (Vectorly custom WebGL) vs **~12% CPU**
(Google MediaPipe Selfie Segmentation, WASM+SIMD) — ~4× lower. The post is explicit that the
win came **mostly from the fully-on-GPU, zero-copy pipeline**, not chiefly a smaller model:
the other libraries *"take data from the CPU, send it to the GPU, and return the result to
the CPU,"* and that round-trip is what Vectorly eliminated. The residual ~3% was mostly just
uploading the video frame to the GPU.

Two levers, very different reach:

| Lever | Size | Status in Longpipe |
|---|---|---|
| **Zero-copy, fully-on-GPU pipeline** | **Big** (the ~4×) | ✅ Already done in **both** WebGL2 **and** WebGPU |
| **Model / compute efficiency** | Smaller, incremental | The lever WebGPU specifically improves over WebGL2 |

**Consequence:** the largest lever is already pulled on every backend. WebGPU's job is the
*second* layer on top — not a repeat of the first 4×.

## 2. What WebGPU adds over the existing WebGL2 zero-copy path

- **Compute shaders + workgroup-shared memory** → real data reuse in conv/GEMM, vs WebGL2's
  fragment-shader-over-textures emulation.
- **Native `f16` (`shader-f16`)** → ~½ memory bandwidth, often ~2× ALU on supporting GPUs.
  The worker already prefers f16 when available.
- **`importExternalTexture`** → can shave even the residual frame-upload cost (most of
  Vectorly's 3%) — so on the **CPU axis, plausibly below 3%**.
- Storage buffers + explicit bind groups + fewer state changes → lower dispatch overhead,
  good fit for the already-fused ops (`Conv2dAdd`, `ProjResidual`, …).

**Expectation:** WebGPU **preserves/exceeds the ~4× CPU win vs MediaPipe** and adds **~1.3–2×
on the model *compute*** — **not** another clean 4× from WebGPU alone. (Estimate from
architecture + WebGPU properties; §5 is how we replace the estimate with a measurement.)

## 3. The gain is use-case-dependent — by bottleneck resource

Each effect bottlenecks on a *different* resource, and WebGPU only accelerates one of them:

| Effect class | Bottleneck | WebGPU gain applies? |
|---|---|---|
| NN video — matting, optical flow, future SR / relight | GPU compute | ✅ **Most** — compute + f16 are exactly this |
| No-model shader/DSP — blur, reframe, low-light, vignette, LUT, overlays, composite | GPU, tiny | ✅ Trivially — already near-free; WebGPU vs WebGL2 ~moot |
| Audio — denoise, VAD, leveler, ducking | **CPU / WASM, audio thread** | ❌ **Irrelevant** — wins come from SIMD / int8 (`dfnint8`), not the GPU |
| Captioning / ASR — Whisper, Moonshine | GPU **or** WASM, separate runtime | ⚠️ Helps, but via transformers.js/ONNX's WebGPU backend, **not** Longpipe's ops |

So "most use-cases" splits cleanly: **visual NN models gain most · no-model effects are
already free · audio gains nothing from WebGPU** (different lever entirely).

## 4. Parallel composition — gains are NOT freely additive

- **GPU work serializes on one queue.** Matting + SR + relight draw from the *same* GPU and
  the *same* per-frame budget (~33 ms @ 30 fps, ~16.6 ms @ 60 fps). Total ≈ Σ each effect's
  GPU time. WebGPU makes each *cheaper* (more room to stack), but stacking multiple **models**
  still consumes the shared budget — and on mid-range/mobile hits **thermal throttling**
  (sustained ≠ burst fps). This is why `skipFrames` + the adaptive controller exist.
- **The genuine free parallelism is *across resources*:** **video-GPU ∥ audio-CPU/WASM** run
  truly concurrently (separate AudioWorklet, separate from the video worker — the architecture
  already enforces this). Matting + denoise compose at near-zero mutual cost.
  Captioning-via-WebGPU, however, **contends** with matting for the GPU.
- **CPU stays low as you stack *shader* effects** — adding blur+reframe+overlay to matting
  barely moves CPU (all on-GPU, zero-copy). That composability is the real inheritance of the
  blog result. Adding an *audio NN* or *ASR* is what adds CPU/GPU back.

**Net:** one NN model + many cheap shader effects + audio denoise compose comfortably and beat
MediaPipe-class CPU. Stacking **multiple heavy models** (matting + SR + relight + ASR) is
budget-bound on weak hardware regardless of WebGPU — it buys headroom, not unlimited stacking.

## 5. How the benchmark harness must prove this (Phase 0.4 spec)

These claims are reasoned estimates — **there is no measured WebGPU-vs-WebGL2 data in the repo
yet.** The bench (Phase 0.4) + autotune microbench should be designed to settle them per
device class:

1. **Per-effect budget (isolated):** model-ms / fps / CPU% for each effect alone, WebGPU vs
   WebGL2, f16 vs f32 — quantify lever #2's real multiplier on this codebase.
2. **Stacked budget (composed):** measure matting → +shader effects → +second model → +audio,
   capturing total GPU-ms vs the frame budget and the **CPU% delta per added effect**
   (verifying shader effects stay ~flat on CPU, audio/ASR add load).
3. **Sustained vs burst:** on mobile, hold the stack for minutes and record throttled fps +
   whether the adaptive controller downgrades correctly.
4. **CPU baseline vs MediaPipe:** reproduce the blog's metric on the same machine — confirm
   ≤3% CPU and quantify how far `importExternalTexture` pushes below it.
5. **Resource attribution:** with the zero-copy audit (roadmap 1.3), label where any remaining
   copy/CPU cost lives, per transport.

Output: committed per-device-class baselines (the §1.3/0.5 regression oracle) and a published
"single vs stacked effect budget" table that turns this note's estimates into numbers.

## 6. Bottom line

- **Individually:** yes — WebGPU matches/exceeds the original gain for the **visual NN models**
  and is trivially fine for **no-model effects**; **audio is a separate lever** (SIMD/quant),
  unaffected by WebGPU.
- **In parallel:** yes for **one model + cheap effects + audio** (the common case);
  budget/thermal-bound when stacking **multiple models**, which is exactly what the adaptive
  controller + presets are for.
- The roadmap already bets correctly: **WebGPU as the durable production core, WebGL2 fallback,
  WebNN as a future seam** — see ROADMAP_EXECUTION Phase 1 / 4 / 8.
