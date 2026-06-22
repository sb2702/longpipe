# Real-time A/V Effects — Competitive Landscape & Catalog

An extensive, sourced catalog of audio/video effects shipped by the major live-video and
editing products, mapped to a **Longpipe disposition** so the roadmap can decide build /
adopt / decline per effect. Compiled June 2026.

**Products surveyed:** Zoom · Microsoft Teams · Google Meet · Apple FaceTime/iOS ·
NVIDIA Broadcast / Maxine · Krisp · Snapchat / TikTok / Instagram · OBS · DaVinci Resolve ·
Adobe Premiere Pro · CapCut. (Live conferencing, creator/streaming, mobile/social, and
editors — to span both real-time and offline expectations.)

## Disposition legend

| Tag | Meaning |
|---|---|
| ✅ **Have** | Already shipped in Longpipe |
| 🟢 **No-model** | Pure shader / DSP / compositing — cheap given the runtime |
| 🟦 **Adopt** | Permissive OSS model exists, real-time-in-browser feasible |
| 🔴 **Train** | Must custom-train — the moat (no permissive real-time-in-browser option) |
| 🧪 **Spike** | Research bet, kill-by-default (see ROADMAP_EXECUTION Appendix B) |
| 🚫 **Decline** | Anti-roadmap — declines the moat / competes with customers / heavy AR |

Cross-reference: model licensing & feasibility detail lives in
[ROADMAP_EXECUTION.md](ROADMAP_EXECUTION.md) Appendices A (video models), B (spikes),
C (audio).

---

## VIDEO EFFECTS

### A. Background / segmentation (Longpipe's core — driven by the matte)

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Background blur (standard) | Zoom, Teams, Meet, FaceTime, NVIDIA | ✅ Have | |
| Background blur (depth-graded / "portrait") | Teams (Portrait blur), FaceTime (Portrait + slider) | 🟢 No-model | Vary blur by a soft alpha falloff; shader on existing matte |
| Virtual background — image | Zoom, Teams, Meet, NVIDIA | ✅ Have | |
| Virtual background — video / animated | Zoom, Teams | ✅ Have | |
| Background replace — solid / greenscreen color | Zoom, NVIDIA | ✅ Have | |
| Background **removal** (transparent alpha out) | NVIDIA, editors (Magic Mask) | 🟢 No-model | Expose the alpha matte as an output for downstream compositing |
| Chroma key / green-screen keying | OBS, editors | 🟢 No-model | Color-space key shader (roadmap *Later*) |
| AI background **generation** (scenes) | CapCut, social | 🚫 Decline | Generative, heavy, not the moat |
| Multi-person / instance matting | editors (Magic Mask multi) | 🔴 Train / 🧪 | Instance-level real-time is hard; *Later* |

### B. Framing / composition

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Auto-framing / face-follow | Meet (Framing), FaceTime (**Center Stage**), NVIDIA (Auto Frame) | 🟦 Adopt / 🟢 | Derive bbox from the matte (no model) or BlazeFace (Apache) |
| Auto-**reframe** to aspect ratio (16:9 ↔ 9:16 ↔ 1:1) | Premiere (Auto Reframe), Resolve (Smart Reframe), CapCut | 🟢 No-model | Crop/track driven by subject box; high value for social repurposing |
| Multi-person gallery framing | NVIDIA, Meet | 🟢 No-model | Compose from per-subject boxes |
| Subject tracking / zoom-follow | NVIDIA, editors (IntelliTrack) | 🟢 No-model | |
| Active-speaker switching | Zoom, Teams, Meet | 🚫 Out of scope | Belongs to the conferencing app, not the effects runtime |

### C. Lighting / color / image quality

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Low-light enhancement / brighten | Zoom (Adjust for low light), Meet (Studio Look), NVIDIA | 🟢 No-model | Exposure/tone-curve shader (Zero-DCE is **non-commercial** — don't use it; see Appx A) |
| Auto exposure / white-balance correction | Meet, editors | 🟢 No-model | Shader, matte-aware (weight the face) |
| Studio / virtual **lighting** (relight, key light) | Meet (Studio Lighting), FaceTime (Studio Light), NVIDIA (Virtual Key Light) | 🧪 Spike B.2 | tier (a) shader light wrap = go; learned relight = research |
| Video **denoise** (sensor noise) | Meet (Studio Look), NVIDIA (Video Noise Removal) | 🟢 No-model / 🟦 | Spatial-temporal denoise shader; matte-guided |
| Sharpening / detail enhance | Meet (Studio Look) | 🟢 No-model | |
| Super-resolution / upscaling | NVIDIA (Super Res), Resolve (Super Scale) | 🟦 Adopt / 🔴 | Light SR feasible; quality SR may need training. *Later* |
| Vignette | NVIDIA | 🟢 No-model | Trivial shader |
| Color filters / LUTs / grading | Zoom, social, editors | 🟢 No-model | LUT shader; "filters" surface |
| HDR / tone mapping | editors | 🟢 No-model | |

### D. Appearance / face touch-up *(careful — the anti-roadmap line runs through here)*

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Skin smoothing / "Touch up" / Soft Focus | Zoom, Teams (Soft Focus), social | 🟢 No-model | Mask-guided bilateral/surface blur — *utility* tier, keep tasteful |
| Face refinement (eyes/teeth/lips detail) | Resolve (Face Refinement) | 🟢 No-model / 🧪 | Needs a light face/landmark mask |
| Teeth whitening / under-eye / blemish | social, YouCam | 🟡 Borderline | Shader-doable; stay on the utility side of the anti-roadmap |
| **Eye-contact / gaze correction** | NVIDIA (Eye Contact), Teams | 🧪 Spike B.1 | No permissive real-time-browser model — research, no-go default |
| Face **reshape** (bigger eyes, slim nose, V-jaw) | TikTok, Instagram, Snap, YouCam | 🚫 Decline | Heavy beauty-AR — perf moat doesn't transfer; Snap/Banuba own it |
| Virtual **makeup** (lip/brow/lashes/foundation) | Zoom (Studio Effects), YouCam | 🚫 Decline (mostly) | Zoom ships light makeup; full try-on is beauty-AR, out of scope |
| Skin-tone / tan alteration | social | 🚫 Decline | Reputational + anti-roadmap |

### E. AR / fun / overlays

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| **Branding overlays** (logo, lower-thirds, name tags, watermark) | Zoom (logo overlay), broadcast | 🟢 No-model | Compositing — roadmap *Next*; B2B2C-friendly |
| Caption / subtitle **rendering** overlay | Meet, Zoom, editors | 🟢 No-model | The *render* layer for Phase 6 captioning |
| Gesture **Reactions** (hearts, confetti, fireworks, balloons) | FaceTime (Reactions), Zoom, Teams | 🚫 Decline | Needs hand/gesture tracking + 3D AR; not the moat |
| AR lenses / masks / 3D objects / face filters | Snap, TikTok, Instagram | 🚫 Decline | Core anti-roadmap (Snap/DeepAR/Banuba territory) |
| Avatars / Memoji / digital personas | Zoom, Teams (Mesh), FaceTime (Memoji) | 🚫 Decline | Rigging/avatar pipeline — adjacency at most (matte+relight primitive) |

### F. Stabilization / motion / restoration (editor-grade, mostly offline)

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Video stabilization | editors, mobile | 🟢 No-model | Motion-based warp; you already have optical flow ops |
| Frame interpolation / smooth slow-mo | Resolve (Speed Warp) | 🔴 Train / 🧪 | Offline-grade; not a live wedge |
| Rotoscoping / **Magic Mask** | Resolve, Premiere | ✅/🔴 | Your live matte *is* the real-time version of this |
| Object removal / content-aware fill / inpainting | Premiere, CapCut | 🚫 Out of scope | Generative/offline; not a real-time browser wedge |

---

## AUDIO EFFECTS

### A. Noise / echo cleanup

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Background noise suppression (stationary + non-stationary) | all | ✅ Have | RNNoise + DeepFilterNet3 |
| Multi-level noise suppression (auto/low/high) | Teams, Zoom | 🟢 No-model wrap | Expose intensity levels over the existing denoiser |
| Acoustic **echo** cancellation (AEC) | Teams, Krisp, NVIDIA | 📌 Rely on browser | `getUserMedia` provides it; coexist, don't rebuild (Appx C) |
| **Room echo / reverb removal** (dereverb) | NVIDIA (Room Echo), Krisp | 🟦 Adopt / 🧪 | DeepFilterNet does some; standalone dereverb = candidate for "audio depth" |
| Wind / keyboard / transient suppression | Krisp, Teams | 🟦 Adopt | Subset of denoise model behavior |
| **Bidirectional** (far-end / inbound) noise removal | Krisp, NVIDIA | 🟢 No-model wrap | Apply the denoiser to the inbound track too |

### B. Voice focus / separation

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| **Voice isolation / target-speaker** (suppress other voices) | Teams (Voice Isolation, enrollment), Resolve | 🧪 Spike B.3 | Needs enrollment; real-time-in-browser unproven |
| Speaker separation / diarization | Krisp, editors | 🚫 Out of scope | Product/transcription layer, heavy |

### C. Enhancement / restoration

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| **Audio super-resolution / bandwidth-extension** (8→16→48 kHz) | Meet (Studio Sound), NVIDIA (Audio Super Res) | 🧪 Spike B.4 | "Studio voice"; learned BWE too heavy for worklet today |
| Speech restoration / de-distortion | NVIDIA, Premiere (Enhance Speech) | 🧪 Spike B.4 | Diffusion/GAN (VoiceFixer) = not real-time-browser |
| Voice clarity / presence / "warmth" / EQ | Krisp, editors | 🟢 No-model | DSP EQ/exciter node |
| De-ess / plosive removal | editors | 🟢 No-model | DSP |

### D. Level / dynamics

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Automatic gain control / auto-leveling | Teams, Meet, browser | 📌 Rely on browser / 🟢 | Browser AGC by default; optional own leveler |
| Loudness normalization / **Dialogue Leveler** | Resolve | 🟢 No-model | DSP — cheap win |
| Compression / limiting | editors, OBS | 🟢 No-model | DSP |
| **Auto-ducking** (music under speech) | Resolve (Auto Ducking) | 🟢 No-model | DSP, needs VAD |

### E. Spatial / creative / passthrough

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| Spatial / 3D audio, stereo widening | FaceTime (Spatial), editors | 🟢 No-model | DSP / Web Audio panner — possible meaning of "audio depth" |
| Reverb add (creative) | OBS, editors | 🟢 No-model | DSP |
| Voice changer / pitch / **Voice Convert** | Resolve, social | 🚫/🧪 | Out of core; novelty |
| **Music mode** / high-fidelity passthrough | Zoom (Original Sound), Teams (Music mode) | 🟢 No-model | Bypass speech processing — needs VAD/music detection |

### F. Speech intelligence (ASR-adjacent)

| Effect | Seen in | Disposition | Notes |
|---|---|---|---|
| **Live captioning / transcription** | Meet, Zoom, Teams, Krisp | 🟦 Adopt | Whisper / Moonshine (MIT) — Phase 6 flagship |
| **VAD / endpointing / smart-mute** | all (implicit) | 🟦 Adopt | Silero VAD (MIT) — new unit 6.0, prerequisite |
| Translation / translated captions | Meet, Teams | 🚫 Out of scope | Product layer; heavier |
| Speaker labeling | Krisp, Teams | 🚫 Out of scope | |
| Summarization / notes / action items | Krisp, Teams, Meet | 🚫 Out of scope | Pure product layer, not an effect |

---

## Synthesis — what this means for Longpipe

1. **The competitive set is ~70% shaders/DSP and compositing, not models.** Most of what
   Meet/Zoom/Teams/NVIDIA ship (blur grades, framing, reframe, low-light, vignette, LUTs,
   leveling, ducking, overlays, captions render) is **🟢 no-model** once you have a good
   matte + a worklet — exactly Longpipe's thesis: *effects are cheap once the runtime
   exists.* These are the fastest adoption wins.

2. **Only a handful require training**, and they're the moat you already invest in:
   real-time **matting quality**, instance/multi-person matting, and (offline-grade)
   super-resolution.

3. **Three research bets**, all kill-by-default: **eye-contact** (B.1), **relighting**
   (B.2), **voice isolation** + **voice BWE** (B.3/B.4). Every major competitor that ships
   these does so with heavier or proprietary models (NVIDIA on RTX, Teams with enrollment),
   which is *why* they're spikes for a browser-MIT runtime, not commitments.

4. **The clear anti-roadmap wall** is the entire mobile/social column — beauty reshape,
   makeup try-on, AR lenses, avatars, reactions. Snap/TikTok/Banuba/DeepAR own it, the perf
   moat doesn't transfer, and chasing it competes with the infra-vendor customers. Decline
   on purpose.

5. **Highest-leverage near-term additions** (no new models, high visible value):
   depth-graded portrait blur · auto-reframe (16:9↔9:16) for social repurposing ·
   low-light/white-balance correction · noise-suppression intensity levels · dialogue
   leveler + auto-duck · branding overlays · caption rendering. Each is a 🟢 unit under the
   Phase 0 gates.

---

## Sources

[Zoom backgrounds/filters/Studio Effects](https://www.zoom.com/en/products/virtual-meetings/features/backgrounds-filters/) ·
[Zoom video enhancements](https://support.zoom.us/hc/en-us/articles/115002595343-Video-enhancements) ·
[Teams video effects / Portrait blur](https://krisp.ai/blog/how-to-blur-background-in-teams/) ·
[Teams noise suppression](https://support.microsoft.com/en-us/teams/meetings/reduce-background-noise-in-microsoft-teams-meetings) ·
[Teams Voice Isolation](https://techcommunity.microsoft.com/blog/microsoftteamsblog/voice-isolation-in-microsoft-teams-enables-personalized-noise-suppression-for-ca/4096077) ·
[Meet Studio Look / Lighting / Framing / Studio Sound](https://support.google.com/meet/answer/13948742) ·
[Meet combine effects / audio](https://workspaceupdates.googleblog.com/2024/01/google-meet-improved-audio-and-video-combine-multiple-video-effects.html) ·
[NVIDIA Broadcast (Eye Contact, Vignette, Virtual Key Light, Auto Frame, Super Res, Room Echo, Audio Super Res)](https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/) ·
[NVIDIA Broadcast 1.4 (Eye Contact + Vignette)](https://www.nvidia.com/en-us/geforce/news/jan-2023-nvidia-broadcast-update/) ·
[NVIDIA Maxine](https://developer.nvidia.com/maxine) ·
[Apple FaceTime video effects (Portrait, Studio Light, Center Stage, Reactions, Memoji)](https://support.apple.com/guide/facetime/use-video-effects-fctm81f99179/mac) ·
[Krisp features](https://krisp.ai/) ·
[Social beauty filters overview](https://en.wikipedia.org/wiki/Filter_(social_media)) ·
[YouCam face reshape/makeup](https://www.perfectcorp.com/consumer/blog/selfie-editing/best-face-filter-apps) ·
[DaVinci Resolve AI (Magic Mask, Smart Reframe, Face Refinement, Voice Isolation, Dialogue Leveler, Auto Ducking)](https://photography.tutsplus.com/articles/davinci-resolve-ai--cms-109186) ·
[CapCut AI (relight, reframe, captions, bg removal)](https://www.capcut.com/tools/relight-videos-with-ai) ·
[Premiere Auto Reframe](https://www.capcut.com/resource/auto-reframe-premiere-pro)
