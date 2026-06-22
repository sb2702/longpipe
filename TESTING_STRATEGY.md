# Cross-Device / Browser Testing Strategy

How to test Longpipe across the device × OS × browser × GPU × backend × transport space —
the detailed build-out of [ROADMAP_EXECUTION.md](ROADMAP_EXECUTION.md) Phase 0.2–0.5.

## The core problem (why generic web testing advice doesn't apply)

Three properties make this SDK hard to test, and they dictate the whole design:

1. **It's GPU shader code.** WGSL/GLSL results differ across GPU vendors and drivers
   (Apple, Adreno, Mali, Intel, NVIDIA, AMD) — rounding, `f16` support, precision, edge
   handling. **A CI runner's software GL/SwiftShader does NOT reproduce real-device
   behavior.** Headless ≠ representative for correctness *or* perf.
2. **It's real-time.** FPS, autotune tier selection, and the adaptive controller only mean
   anything on real hardware with a real thermal envelope. Perf numbers from a VM are noise.
3. **It needs live media + permissions + a secure context.** `getUserMedia`, a video track,
   mic, `AudioContext` user-gesture rules, COOP/COEP for SIMD — all must be driven
   deterministically.

**Design consequence — split the axes by what they actually need:**

| Tested in software CI (fast, deterministic) | Needs real hardware (nightly/release) |
|---|---|
| Op/block/network numeric parity vs PyTorch | Shader correctness on real GPUs (Apple/Adreno/Mali/…) |
| Cross-backend parity (WebGPU≡WebGL) logic | Real-time FPS + autotune tier selection |
| Transport **selection** logic, capability laddering | Thermal/soak behavior on mobile |
| Fallback laddering with mocked capabilities | `f16` availability + numeric behavior per device |
| API surface, error paths, message protocol | End-to-end matte quality on real silicon |

Don't try to prove perf or GPU correctness in headless CI, and don't burn real-device
minutes on logic a mock can cover. Most regressions are caught cheaply on the left; the
right is the irreplaceable confidence layer.

---

## 1. Test taxonomy (levels) and where each runs

```
L0 unit/logic        ── PR gate (Node)         topology select, preset math, weight pack, normalizers
L1 op/block/network  ── PR gate (Playwright    fixtures vs PyTorch ≤1e-4; cross-backend parity
                        Chromium, headed+GPU)
L2 pipeline e2e      ── PR gate + nightly       golden input clip → golden output (perceptual tol)
L3 capability/fallback ─ PR gate                force each WebGPU→WebGL2→WASM rung; assert frames flow
L4 device matrix smoke ─ nightly (real devices) "correct matte at expected tier" per cell
L5 performance/budget ─ nightly (real devices)  model-ms/fps/tier vs per-device-class baseline
L6 soak/memory       ── nightly (real devices)  long session: flat heap + bounded GPU buffers
L7 release sweep     ── pre-publish             full matrix incl. old/low-end + manual spot-check
```

L0–L3 are the fast gate (every PR, minutes). L4–L6 are the real-device confidence layer
(nightly + on-demand). L7 gates `npm publish`.

---

## 2. The device / browser matrix (risk-prioritized, not combinatorial)

Test **device *classes*** (what autotune actually distinguishes), not every SKU. The full
cross-product is thousands of cells; this is the ~dozen that matter.

### Tier 1 — must always be green (the bulk of users + the hardest engines)

| Class | OS | Browser | Backend(s) | Why it's here |
|---|---|---|---|---|
| Apple Silicon Mac | macOS | **Safari** | WebGPU + WebGL2 | WebKit engine; Apple GPU; can't be faked on Linux |
| Apple Silicon Mac | macOS | Chrome | WebGPU (f16) | Primary dev target; f16 path |
| Windows + dGPU (NVIDIA/AMD) | Win | Chrome/Edge | WebGPU + WebGL2 | Largest desktop slice |
| Windows iGPU (Intel) | Win | Chrome | WebGPU/WebGL2 | Common low-end desktop |
| Windows | Win | **Firefox** | WebGL2 (+WebGPU as it lands) | Gecko; **the `transfer-capture` transport quirk** (`topology.ts` UA path) |
| iPhone (recent) | iOS | Safari | WebGPU(by ver)/WebGL2 | Hardest cell; mobile WebKit + thermal |
| Mid-range Android | Android | Chrome | WebGL2/WASM | **Roadmap's explicit floor** (Mali/older Adreno) |

### Tier 2 — nightly / release sweep

High-end Android (Snapdragon/Adreno, WebGPU) · iPad · ChromeOS · **netbook-class / 10-yr-old
GPU** (the README's WebGL2/WASM floor claim — must be *proven*) · older Safari (N-1) · Android
WebView / RN target (Phase 8 seam).

### Axes folded into each cell (via config injection, §4)
`backend ∈ {webgpu, webgl}` · `dtype ∈ {f16, f32}` · `preset ∈ {auto, fast, balanced, quality}` ·
`transport` (the browser picks; force-override to exercise others) · `audio ∈ {off, denoise}`.
One physical device covers many code paths by re-running with forced configs.

---

## 3. How to obtain the environments (the procurement reality)

1. **Playwright multi-engine (local + PR CI)** — Chromium, Firefox, WebKit. **Caveat:**
   Playwright-WebKit on Linux is *not* real Safari/iOS and renders with software GL — good
   for **logic**, useless for **GPU correctness/perf**. Treat it as an engine-logic check,
   not a Safari proxy.
2. **Self-hosted runners (the GPU truth layer)** — minimum viable lab:
   - 1× **Apple Silicon Mac mini** → real Safari + Chrome WebGPU on Apple GPU (also drives a
     tethered **iPhone/iPad** via Playwright/WebDriver).
   - 1× **Windows + discrete GPU** box → Chrome/Edge/Firefox WebGPU+WebGL2 on real drivers.
   - 1× **physical mid-range Android** (e.g. a Mali-class phone) → the roadmap floor.
3. **Cloud device farm (breadth)** — BrowserStack / LambdaTest / Sauce for real Safari, iOS,
   Android, and Windows-GPU breadth without buying every device. **The catch you must verify
   before relying on it:** many cloud desktop VMs expose **SwiftShader/software GL**, which
   gives *wrong* perf and can mask/introduce shader bugs. Confirm real GPU + WebGPU adapter
   on each provider/cell (log `adapter.info` / `WEBGL_debug_renderer_info`); use real **mobile
   devices** (which do have real GPUs) for the mobile cells, and self-hosted boxes for desktop
   GPU truth.

**Recommendation:** Playwright everywhere as the automation layer + a **small self-hosted lab
(Mac mini + Windows-GPU + one Android)** for GPU/perf truth + **one cloud farm** for Safari/iOS
breadth and old-device sweeps. This is the cheapest path to trustworthy coverage. Decision
needed: which cloud provider, and self-host-vs-rent the Windows-GPU box (see Open Decisions).

---

## 4. Deterministic input + the test harness

**Synthetic media (no real webcam, fully reproducible).** Feed a known clip so output is
comparable across runs/devices:
- Chromium/Edge: `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=clip.y4m`
  `--use-fake-ui-for-media-stream` (auto-grant) `--autoplay-policy=no-user-gesture-required`;
  WebGPU sometimes needs `--enable-unsafe-webgpu` / a Vulkan/Metal flag.
- Firefox: `media.navigator.streams.fake` + permission prefs.
- Serve the harness over **https/localhost** (secure context required by `getUserMedia`).

**Golden input set** (committed or in a fixtures bucket) — cover the matte's hard cases, not
just an easy headshot: `single_person`, `multi_person`, `low_light`, `backlit`, `fast_motion`,
`hair_detail`, `no_subject`. A matching mic clip for audio.

**Harness page** — a stripped build of `demo/` that the automation drives, exposing hooks:
```js
window.__lp = {
  start(config),                 // force backend/dtype/preset/transport/audio
  ready: Promise,                // resolves on first effect frame
  captureFrame(): ImageData,     // for golden/visual compare
  stats(): RendererStats,        // fps, modelMs, resolved tier, fallback rung
  run(nFrames): Promise<Summary> // batch for perf/soak
}
```
This is the single seam every L2–L6 test drives, on every device.

---

## 5. Correctness assertion (GPUs differ — don't assert bit-exactness end-to-end)

Use the **tightest tolerance the level can bear**:

- **L1 op/block (per backend):** vs PyTorch fixtures. `f32 ≤ 1e-4`. **`f16` needs a separate,
  looser bar** (relative, or alpha-error based) — bit-exactness across f16 implementations is
  unrealistic. Cross-backend parity asserts WebGPU≡WebGL within the same tolerance class.
- **L2 end-to-end (per device):** golden input → assert the **matte quality metric** (alpha
  **MAE / IoU** vs a reference matte) and a **perceptual** image compare (SSIM/MAE band), NOT
  pixel-exact — Apple vs Adreno vs Intel will differ by a few LSBs legitimately. Set the band
  per device-class baseline; flag drift, don't fail on rounding.
- **Visual regression (L2):** per-device baseline snapshots with a tolerance mask; new
  snapshots reviewed on first sight (Phase 0.7 parity runner generates them).
- **L4 smoke (per real device):** cheapest possible — "the matte is non-blank, roughly
  correct (IoU > floor), at the expected tier, no crash." This is breadth coverage, not
  precision.

---

## 6. Performance & soak (real hardware only)

- **L5 perf:** the Phase 0.4 benchmark harness run via `__lp.run()` on each real device-class.
  Capture model-ms / fps / **autotune-selected tier** / memory; store a committed
  **per-device-class baseline**; fail on >X% regression. Mid-range Android must land a
  30fps-capable tier under `auto` (roadmap 1.5).
- **Thermal/sustained:** mobile throttles — measure **sustained** fps over minutes, not a
  burst, and assert the adaptive controller *downgrades* correctly under thermal pressure.
- **L6 soak:** long session via `__lp.run(largeN)` asserting **flat JS heap + bounded GPU
  buffer count** (Phase 0.5). Run nightly on at least one desktop + one mobile class.

---

## 7. Capability / fallback / transport coverage (mostly software-testable)

- **Force each rung** (L3): mock `navigator.gpu` absent → WebGL2; force `shader-f16` off →
  f32; simulate `device.lost` / `webglcontextlost` → assert rebuild-and-resume (roadmap 1.2);
  disable WASM-SIMD → RNNoise fallback. Each must **keep emitting frames**, never hard-fail.
- **Transport matrix:** assert `selectTopology()` picks the right pair per engine, and
  force-override to run the renderer through **all 6 input×output combos** (the renderer is
  identical across them — prove it). This is the place to kill the current UA-sniffing in
  `topology.ts` in favor of feature probes (roadmap 1.1).
- **Audio:** sample-rate conversion paths, SIMD-present vs absent, `AudioContext` resume on
  gesture, denoise passthrough-until-ready.

---

## 8. CI tiering (cost × speed)

| Stage | Trigger | Runs | Target |
|---|---|---|---|
| **PR gate** | every push | typecheck · lint · build · L0 (Node) · L1–L3 (Playwright Chromium, headed+GPU) | < ~10 min; blocks merge |
| **Engine check** | every push | L1/L2 logic on Playwright Firefox + WebKit (logic only) | catch engine-specific JS breaks |
| **Nightly** | schedule | L4 smoke + L2 golden + L5 perf + L6 soak on the **real-device matrix** (self-hosted + cloud) | regenerates `BROWSER_MATRIX.md`, perf baselines |
| **Release sweep** | pre-`publish` | L7 full matrix incl. Tier 2 / old devices + manual spot-check | gates `npm publish` |

GitHub Actions (or equiv) orchestrates; self-hosted runners join as labeled runners; the
cloud farm is invoked from the nightly job. `prepublishOnly` already builds — extend the
release path to require the sweep.

---

## 9. Living artifacts

- **`docs/BROWSER_MATRIX.md`** — the support statement, **regenerated by the nightly job**:
  green / known-gap / unsupported per cell, with the failing-rung note for gaps. This is what
  you point customers at, and it makes "works everywhere" auditable.
- **Per-device perf baselines** (committed JSON) — the regression oracle for L5.
- **Golden assets** — input clips + reference mattes, versioned with the model weights.
- **Field telemetry as the infinite matrix** (Phase 7) — opt-in, media-free FPS / fallback-rung
  / tier distribution from real users catches the long tail of devices the lab will never own.
  The lab proves the cells you chose; telemetry tells you which cells you forgot.

---

## 10. Build order (don't boil the ocean)

1. **Harness page + synthetic-media flags + `__lp` hooks** — nothing else works without a
   driveable, deterministic target. Land first.
2. **PR gate** (L0–L3 on Playwright Chromium) + **golden L2** on one device — the everyday
   safety net.
3. **Parity runner + visual snapshots** (Phase 0.7) — makes adding any op auto-covered.
4. **One self-hosted Mac mini** → real Safari + Apple-GPU WebGPU + tethered iPhone. Biggest
   single confidence jump (the hardest cells).
5. **Cloud farm** for iOS/Android/old-desktop breadth → first `BROWSER_MATRIX.md`.
6. **Perf baselines + soak** on the lab → wire regression gates.
7. **Windows-GPU + Android self-hosted** → close Tier 1; **release sweep** → gate publish.

> Each step is independently valuable: even step 2 alone turns "we think it works" into
> "logic + parity are provably green on every PR." The device lab then upgrades that to
> "provably works on the silicon our users actually run."
