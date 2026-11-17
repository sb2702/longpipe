# End-to-end smoke tests

Playwright Test drives [`examples/vanilla-webrtc`](../../examples/vanilla-webrtc) in Chromium,
Firefox and WebKit. Each test puts `pipeline.stream` through a real `RTCPeerConnection` loopback
(pc1 → pc2) and checks what **pc2 received**.

```bash
npm run test:e2e                         # all browsers; starts the example server itself
npm run test:e2e -- --project=firefox    # one browser
npx playwright show-report tests/e2e/results/html
```

Output in `tests/e2e/results/` (gitignored): **`summary.md`** (the matrix: scenario × backend
requested → per browser pass/skip/FAIL and the backend actually used, with skip/fail reasons),
`summary.json`, `results.json`, and the HTML report. Traces and screenshots are kept for failures.

## Input (no fake cameras)

`npm run e2e:assets` ([`fetch_assets.mjs`](fetch_assets.mjs), also run by the test web server)
downloads, by pinned URL + sha256 like `benchmarks/performance`, into
`examples/vanilla-webrtc/public/assets/` (gitignored, nothing third-party committed):

- the 640×360 clip of a person (the benchmark's clip). The page draws it into a `<canvas>` and
  uses `canvas.captureStream()`, which works in every engine, Safari included;
- `model_xs` weights (f16 + f32) and `rnnoise.wasm`. The tests pin `preset: 'fast'` (xs, quick on
  software GPUs) and denoise model `rnnoise`, and load both locally (`weights=local`), so runs are
  reproducible offline and don't hit cdn.longpipe.dev (its logs feed adoption metrics).

Audio is WebAudio: a 440 Hz tone + white noise into a `MediaStreamAudioDestinationNode`. There's
no speech, so a working denoiser cuts it by ~40 dB (0.15 RMS → ~0.001). That's the check.

## Scenarios

Each of the first five runs with `backend: 'auto'`, `'webgpu'` and `'webgl2'` (the SDK option). If
a forced backend isn't available the SDK falls back (as it should) and the test is **skipped**,
with the reason, rather than passing on the wrong backend.

| Scenario | Checks |
|---|---|
| background only | remote `framesDecoded` keeps rising; corners of the remote frame are the chosen green (±60), face/chest aren't; status `running`, no fatal errors |
| audio only, denoise on | remote audio samples rising and not silent; SDK reports the kernel `active` and the tone+noise is attenuated, **or** it reports the fallback (`audio` error, non-fatal) and the original audio passes through |
| background + audio | both |
| fail-open: unreachable weights | `weightsBaseUrl` = a refused port. `ready` rejects with a fatal `init-failed` (or `recovery-failed`); remote frames keep flowing **raw** (corners not green); denoise reports `audio` and the original audio flows |
| fail-open: GPU device lost mid-call | `_debug('lose-backend')` (WebGPU `device.destroy()` / `WEBGL_lose_context`): non-fatal `backend-lost`, frames keep reaching pc2, `onRecovered` fires, the green background comes back. On Safari the output track is replaced, and the example's `onTrackChanged → replaceTrack` keeps the remote video going |
| fail-open: no GPU | `backend: 'none'`: `ready` resolves with backend `none`, non-fatal `no-backend`, raw frames flow, `getStats()` says `none`/`passthrough` |
| getUserMedia fake camera | bonus: the browser's fake camera (Chromium, Firefox) through the same call |

Every test ends with the example's Stop (`pipeline.destroy()`, close the call, stop the input):
no console errors in the 1.5 s after, and both output tracks `ended`.

## Running headless on Linux: what works (Nov 2026, Playwright 1.59)

| | WebGPU | WebGL2 (in the SDK's worker) | Notes |
|---|---|---|---|
| Chromium 147 | adapter with `--enable-unsafe-webgpu` (SwiftShader, no `shader-f16`); with `E2E_CHROMIUM_GPU=vulkan` the real GPU (Intel, f16). **But** the device is destroyed (`reason: 'destroyed'`) the first time anything is rendered to a WebGPU canvas, on any thread and with both adapters; Chrome 154 too. So "webgpu" is skipped and "auto" ends on WebGL2 after two losses | SwiftShader | output ~10-12 fps at xs on SwiftShader |
| Firefox 148 | `navigator.gpu` exists with `dom.webgpu.enabled`, but `requestAdapter` fails: "WebGPU is disabled by blocklist" | yes (host GPU locally, Mesa on CI with `webgl.force-enabled`) | |
| WebKit (WPE) 26.4 | no `navigator.gpu` | **none**: no WebGL2 on `OffscreenCanvas` in workers | only the no-GPU path and audio can be tested; see below |

Minimal repro of the Chromium WebGPU loss (no SDK): request a device, `configure()` a canvas, do
one render pass to `getCurrentTexture()`, submit; `device.lost` resolves with `destroyed` within
~10 ms. Longpipe's WebGPU path works in normal (headed) Chrome, so this looks like a headless
limitation rather than an SDK problem; it wasn't checked headed here (that would open windows on
the desktop session). It did expose an SDK bug (fixed): a device lost before `startRender`
wasn't counted, so the SDK retried WebGPU until recovery gave up instead of moving to WebGL2.

### WebKit on a machine without its system libraries

Playwright's WebKit needs `libavif13` etc. With sudo: `sudo npx playwright install-deps webkit`
(CI does this). Without sudo (Ubuntu 22.04):

```bash
eval "$(tests/e2e/tools/webkit-libs-nosudo.sh)"   # apt-get download + unpack + LD_PRELOAD
npm run test:e2e -- --project=webkit
```

### WebKit audio

Playwright's WebKit sends the **upstream** audio of a WebAudio-processed track over WebRTC: a
`MediaStreamAudioDestinationNode` track that's attenuated 40 dB locally arrives at pc2 at full
level. This reproduces without Longpipe (tone → `MediaStreamSource` → gain 0.01 → destination →
`addTrack`: Chromium receives 0.0014 RMS, WebKit 0.14). So on WebKit the remote attenuation is
recorded in the summary but not asserted; `pipeline.stream`'s own audio is. **Check on real
Safari** (manual checklist step 3 in the example README): if real Safari does the same, the
denoiser has no effect on what Safari callers send.

### Known flakiness

WPE WebKit's web process occasionally crashes ("Target crashed") during pipeline startup when the
worker probes for a GPU (~1-2 of 17 tests per run here; never with `backend: 'none'`). CI retries
once.

## CI

[`.github/workflows/e2e.yml`](../../.github/workflows/e2e.yml) (draft): ubuntu-latest, Node 22,
`playwright install --with-deps`, cached assets, the suite, the HTML report + summary as artifacts,
and `summary.md` in the job summary. It runs on software GPUs only: WebGL2 effects on Chromium and
Firefox, no WebGPU anywhere, no real Safari/iOS. A commented BrowserStack outline is at the bottom
for real Safari/iOS later.
