# Backlog — No-Model Effects

Concrete, PR-sized units for the high-leverage effects that need **no new ML model** —
pure shader / DSP / compositing on top of the existing matte + AudioWorklet. These are the
fastest adoption wins (cf. [EFFECTS_LANDSCAPE.md](EFFECTS_LANDSCAPE.md) finding #5).

Every unit inherits the **Phase 0 gates** and the operating model from
[ROADMAP_EXECUTION.md](ROADMAP_EXECUTION.md) §0 (Definition of Done): dual-backend in one
PR · WebGPU≡WebGL parity (`≤1e-4`) · per-frame budget green · matrix green or documented ·
public API typed + minimal · docs updated. "No-model" is itself an acceptance criterion:
**no weights are fetched and `pipeline.ready` timing is unchanged.**

Ticket template: **Scope · Where · Depends · Acceptance · Tests · Budget · API delta · Effort/Priority.**

| ID | Effect | Surface | Effort | Priority | Depends |
|----|--------|---------|--------|----------|---------|
| NM-1 | Depth-graded portrait blur | Video | S | P0 | — |
| NM-2 | Auto-reframe (aspect-ratio) | Video | L | P0 | subject-bbox infra |
| NM-3 | Low-light / white-balance correction | Video | M | P0 | — |
| NM-4 | Noise-suppression intensity levels | Audio | S | P0 | — |
| NM-5 | Dialogue leveler + auto-duck | Audio | M | P1 | NM-5a (VAD, unit 6.0) for duck |
| NM-6 | Branding overlays (logo / lower-third) | Video | M | P1 | — |
| NM-7 | Caption rendering layer | Video | M | P1 | Phase 6 (content) |

> **Shared prerequisite — subject bbox.** NM-2 (and, cheaply, auto-framing) needs a
> temporally-smoothed subject bounding box derived from the alpha matte. Build it once as a
> small GPU reduction (or downsampled-alpha readback) + EMA smoother and reuse it. Track as
> **NM-0** if you want it as its own unit.

---

## NM-1 — Depth-graded portrait blur

- **Scope.** Blur that ramps with distance from the subject (bokeh-like falloff) instead of
  a single uniform sigma over the whole background — matches Teams "Portrait blur" /
  FaceTime Portrait. No depth model: grade the blur by distance from the matte edge (a
  signed-distance / dilated-alpha falloff drives a per-pixel sigma or a mip-level blend).
- **Where.** `src/model/effects/compositor_blur.ts`, `src/model/effects/blur_pyramid.ts`;
  add a `grade` param to `BackgroundConfig.blur` in `src/model/render_op.ts` +
  `src/pipeline/background.ts`. The blur pyramid already exists — blend pyramid levels by
  the falloff rather than picking one.
- **Depends.** None.
- **Acceptance.**
  - `background: { blur: { sigma, grade?: 'none' | number } }` — `grade` controls falloff
    width; `grade:'none'` reproduces today's uniform blur exactly (regression-safe default).
  - Smooth gradient, **no banding** on a flat background; no haloing at the matte edge.
  - Runtime-swappable via `setBackground` like other blur configs.
- **Tests.** New fixture comparing graded output to a reference at 2–3 falloff widths;
  parity WebGPU≡WebGL; visual-regression snapshot (Phase 0.7); `grade:'none'` byte-identical
  to current blur.
- **Budget.** ≤ current blur per-frame cost + small constant (reuses the existing pyramid).
- **API delta.** Additive `grade?` field; no breaking change.
- **Effort/Priority.** S / P0.

## NM-2 — Auto-reframe (aspect-ratio reframe + subject tracking)

- **Scope.** Keep the subject framed (headroom-correct, centered) while emitting a chosen
  output aspect ratio — the 16:9 ↔ 9:16 ↔ 1:1 social-repurposing lever (Premiere Auto
  Reframe / Resolve Smart Reframe / FaceTime Center Stage). Crop+pan+zoom driven by the
  smoothed subject bbox; no model.
- **Where.** New module computing the bbox from the alpha matte (see NM-0) +
  a smoothing/easing controller; apply the crop in the output path
  (`src/pipeline/worker/renderer.ts` / output sizing in `setup_output.ts` /
  `pickOutputSize` in `pipeline/index.ts`).
- **Depends.** Subject-bbox infra (NM-0).
- **Acceptance.**
  - `reframe: { aspect: '9:16' | '16:9' | '1:1' | number, headroom?, padding? }`.
  - Subject stays framed with stable headroom; **no jitter** — motion is eased (peak-hold
    attack / slow release), no per-frame snapping.
  - Recovers gracefully when the subject leaves/re-enters frame; falls back to center-crop
    when no subject is detected.
  - Composes with backgrounds (reframe happens on the composited output).
- **Tests.** Scripted bbox track (synthetic subject motion) asserting framing + smoothing;
  cross-browser matrix (geometry only, but verify transport interaction); no-flicker metric.
- **Budget.** bbox reduction + crop ≤ small fixed cost; must not add a readback stall on the
  per-frame path (compute bbox at model FPS, not every display frame).
- **API delta.** New `reframe` option + `setReframe()` runtime control.
- **Effort/Priority.** L / P0 (highest external value — unlocks social repurposing).

## NM-3 — Low-light / white-balance correction

- **Scope.** Matte-aware exposure + tone-curve brightening and auto white-balance, as a
  shader (Meet "Studio Look" lighting-fix, NVIDIA Video Noise Removal's low-light intent).
  **No model** — Zero-DCE is non-commercial (Appendix A), so this is an explicit shader
  path, with a temporally-smoothed auto-gain estimated from the luma histogram (face-weighted
  via the matte).
- **Where.** New op `src/model/effects/` (e.g. `tone_correct`) + WGSL/GLSL shaders, inserted
  in the render path before composite; auto-gain state on the renderer.
- **Depends.** None (can reuse the matte for face-weighting; works without it too).
- **Acceptance.**
  - `enhance: { lowLight?: 'auto' | number, whiteBalance?: 'auto' | 'off' }`.
  - Brightens shadows without clipping highlights; **no frame-to-frame flicker** (gain is
    smoothed); converges within ~0.5 s on a lighting change.
  - `auto` adapts to scene luma; manual value is respected and never auto-overridden.
- **Tests.** Fixtures at low/normal/backlit exposure asserting target luma + no clipping;
  temporal-stability test (constant input → constant gain); parity.
- **Budget.** One full-frame shader pass; small budget line declared in the bench.
- **API delta.** New `enhance` option + `setEnhance()`.
- **Effort/Priority.** M / P0.

## NM-4 — Noise-suppression intensity levels

- **Scope.** Expose Auto / Low / High denoise levels (like Teams/Zoom) over the **existing**
  denoiser — no new model. "High" = current full denoise; "Low" = dry/wet mix or lighter
  post-filter; "Auto" = current behavior. Pure wrapping/mixing in the worklet.
- **Where.** `src/audio/denoiser.ts` + worklet processor (`src/audio/worklet/processor.ts`);
  options in `src/pipeline/audio.ts` + public `DenoiseOptions`.
- **Depends.** None.
- **Acceptance.**
  - `audio: { denoise: { level: 'auto' | 'low' | 'high' } }` and `setDenoise({ level })` at
    runtime, glitch-free on switch (cross-fade, no click).
  - `getAudioStats()` reflects the active level.
  - Backwards-compatible: existing `model`/`enabled` semantics unchanged.
- **Tests.** Worklet unit test asserting wet/dry mix per level; click-free transition check;
  stats round-trip.
- **Budget.** Within the ~2.7 ms worklet quantum (mixing is trivial).
- **API delta.** Additive `level?` on `DenoiseOptions`.
- **Effort/Priority.** S / P0.

## NM-5 — Dialogue leveler + auto-duck

- **Scope.** Two DSP nodes in the worklet chain: **(a) leveler** — slow AGC toward a target
  loudness (Resolve Dialogue Leveler) without pumping; **(b) auto-duck** — attenuate a
  background/music track when speech is present (Resolve Auto Ducking). No model; ducking
  needs a speech gate.
- **Where.** New DSP in `src/audio/` worklet chain (compose after denoise).
- **Depends.** Auto-duck depends on **NM-5a = VAD (unit 6.0, Silero MIT)** for the speech
  gate. The leveler ships independently of VAD.
- **Acceptance.**
  - Leveler holds target LUFS within tolerance; smooth attack/release, **no pumping** on
    speech pauses.
  - Auto-duck attenuates the secondary track by a configurable amount with smooth
    attack/release, keyed off VAD; releases cleanly when speech stops.
  - Both individually toggleable; off by default.
- **Tests.** Offline DSP tests on scripted loud/quiet + speech/silence buffers asserting
  target loudness and duck envelope; CPU-budget assertion.
- **Budget.** Within the worklet quantum.
- **API delta.** New audio options (`leveler`, `duck`) + runtime setters; VAD wiring.
- **Effort/Priority.** M / P1 (leveler P0-able if VAD slips).

## NM-6 — Branding overlays (logo / lower-third)

- **Scope.** Composite a logo image and/or a lower-third text band over the output
  (Zoom logo overlay; broadcast lower-thirds) — the B2B2C-friendly branding surface. Pure
  compositing presenter; no model.
- **Where.** New presenter/op `src/model/backends/{webgpu,webgl}/ops/composite_overlay.*`
  (+ shaders) registered in both backend `index.ts` `presenters`; wired through
  `render_op.ts` as a final compositing pass; pipeline `overlay` option +
  `setOverlay()`. Honors `RenderTarget` (main + preview).
- **Depends.** None.
- **Acceptance.**
  - `overlay: { image?, text?, anchor, margin, opacity, scale }`; pixel-correct placement at
    each anchor; opacity respected.
  - Survives background swaps and `setEnabled` toggles; renders on the preview canvas too.
  - Multiple overlays (logo + lower-third) compose in order.
- **Tests.** Snapshot per anchor/opacity; parity WebGPU≡WebGL; preview-path render check.
- **Budget.** One small compositing pass; negligible.
- **API delta.** New `overlay` option + `setOverlay()`/`clearOverlay()`.
- **Effort/Priority.** M / P1.

## NM-7 — Caption rendering layer

- **Scope.** The **render** surface for captions (the model is the ASR, delivered later) —
  draw timed, styled caption text with a legible scrim over the output. Landable *before*
  Phase 6 against a scripted/mock caption feed, so the visual layer is done and verified
  independently of ASR.
- **Where.** Text render in the worker over the output (textured-quad glyphs or an offscreen
  2D canvas composited in); caption-track input over the control channel
  (`pipeline/messages.ts` + `worker/index.ts` handler); style config.
- **Depends.** Phase 6 for live content; **no dependency for the render layer itself**
  (drive it with a scripted track in tests/demo).
- **Acceptance.**
  - Legible captions with background scrim; configurable position/size/max-lines; correct
    word/line timing from the supplied track.
  - Renders identically across both backends and all transports; **no per-frame allocation**
    in steady state (Phase 0.5 soak-safe).
  - Graceful with empty/rapid caption updates.
- **Tests.** Scripted caption track → snapshot at known timestamps; soak (no leak); transport
  matrix.
- **Budget.** Text raster cached per caption line (not per frame); composite pass small.
- **API delta.** `captions` option + `pushCaption()/setCaptionStyle()` (or internal hook
  consumed by Phase 6).
- **Effort/Priority.** M / P1.

---

## Additional trivial wins (no template needed — same gates)

Each is a single shader/DSP node, mostly an afternoon once NM-1/3/6 establish the patterns:

- **Vignette** (shader) · **Color filters / LUTs** (LUT shader, the "filters" surface) ·
  **Alpha-out / background-removal output** (expose the matte as a transparent output) ·
  **Tasteful skin smoothing** (matte+face-mask guided surface blur — *keep on the utility
  side of the anti-roadmap*) · **Stereo widen / spatial pan** (Web Audio, a candidate
  concrete meaning for "audio depth") · **Music-mode passthrough** (bypass speech processing,
  VAD/music-gated).

## Suggested order

1. **NM-1, NM-3, NM-4** (all S/M, no deps) — immediate visible wins, establish shader +
   audio-wrap patterns.
2. **NM-0 → NM-2** — bbox infra then auto-reframe (highest external value; gate it behind
   the bbox unit).
3. **NM-6** (overlays) — B2B2C surface; **NM-5 leveler** (VAD-independent half).
4. **NM-7 caption render** + **NM-5 auto-duck** — as VAD (6.0) and Phase 6 land.
