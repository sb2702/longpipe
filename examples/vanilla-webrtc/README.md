# Longpipe + vanilla WebRTC

The smallest complete example of Longpipe in a call:

```
getUserMedia (video + audio)
  → new EffectsPipeline(stream, { background, audio: { denoise } })
  → pipeline.stream          (processed video + denoised audio, live immediately)
  → pc1.addTrack(...)        RTCPeerConnection loopback in the page
  → pc2 → remote <video>     what the other side would see and hear
```

The Longpipe-specific code is `startPipeline()` in [`main.js`](main.js). [`call.js`](call.js) is plain
WebRTC (two peer connections with in-page signaling, plus the few RTP stats worth watching).
[`inputs.js`](inputs.js) has the camera input and a camera-free test input.

What it shows:

- **Processed tracks in a real `RTCPeerConnection`**: the remote `<video>` is what pc2 decoded.
- **Controls**: background none / blur / image / solid colour (live `setBackground`), audio denoise
  on/off (live `setDenoise`), and backend auto / prefer WebGPU / force WebGL2 / none (a new
  pipeline, with the tracks swapped on the existing senders via `replaceTrack`).
- **Model**: `auto` (benchmark the device, then the adaptive controller moves between tiers), a
  fixed tier `xs` / `small` / `medium` / `large` / `xl` (`preset: PRESETS[i]`; switching between
  tiers is live, `pipeline.setPreset()`, same track, and it stops the adaptive controller), or
  `MediaPipe selfie` (`model: 'mediapipe-selfie'`). `auto` and MediaPipe are construction options,
  so switching to or from them destroys the pipeline and builds a new one, tracks swapped with
  `replaceTrack`.
- **Bypass Longpipe**: the raw camera + mic tracks go straight onto pc1's senders (`replaceTrack`)
  and the pipeline is destroyed, so nothing competes for the CPU/GPU: the WebRTC-only baseline.
  Unticking builds a new pipeline and swaps its tracks back. No renegotiation either way.
- **Live readout**: what's actually running (model · dtype · backend), input and output fps,
  frames sent/decoded and remote fps from `getStats()`, received audio, denoise state, transport
  paths, restarts, last error, and a **latency breakdown** (below).
- **`onTrackChanged` → `sender.replaceTrack(videoTrack)`**: the canonical handler. It only fires on
  Safari after its worker died (the output track had to be replaced); everywhere else the output
  track never changes.
- **Pitfall avoided**: an `EffectsPipeline` is a thenable (`await pipeline` waits for `ready`).
  Don't return one from an `async` function or `await` it unless you mean to wait for the effect;
  keep it in a variable and use `pipeline.ready`.

## Run it

From the repo root:

```bash
npm install
npm run e2e:assets   # optional: the test clip + local weights (needed for "test clip" input / weights=local)
npm run example      # http://localhost:5287
```

Click **Start**. "camera + mic" needs a camera; "test clip + tone/noise" doesn't (it loops a
clip of a person into a canvas and synthesises a 440 Hz tone plus noise, which the denoiser should
almost entirely remove: there's no speech in it).

URL parameters preset the controls, which is handy on a phone:
`?input=clip&bg=color&color=%2300ff00&denoise=on&backend=webgl2&model=large&bypass=1&flash=1`
(`model=auto|xs|small|medium|large|xl|mediapipe`). More have no control:
`preset=fast|balanced|quality` (older spelling of `model=xs|medium|xl`), `camera=front|back`,
`weights=local` (load weights from `./assets/models/` instead of the CDN; `npm run e2e:assets`
only fetches `xs`, so other tiers and MediaPipe need the CDN) and `debug=1` (verbose SDK logging).

### Reading the latency numbers

Each row is per frame, averaged over the last second (deltas of the cumulative counters, so a
model switch or bypass shows up within a second):

| Row | Source |
|---|---|
| capture | ≈ one frame interval at the input frame rate (not measured) |
| Longpipe | `pipeline.getStats().latencyMs`: input frame entering the pipeline → its output frame handed to the output transport, measured in the worker. On Safari/Firefox (`rvfc-postmessage`) the clock starts when the SDK grabs the frame on the main thread; on Chrome (`mstp`) when the worker takes it. GPU completion and the browser's canvas capture come after, so it is a lower bound |
| encode | pc1 `outbound-rtp` `totalEncodeTime / framesEncoded` |
| jitter buffer | pc2 `inbound-rtp` `jitterBufferDelay / jitterBufferEmittedCount` (time from a frame's first packet to it leaving the buffer for decode) |
| decode | pc2 `inbound-rtp` `totalDecodeTime / framesDecoded` |
| estimated end to end | the sum. **An estimate**: it leaves out the network (≈ 0 in a loopback; add your RTT/2 for a real call), packetization and display (≈ one screen refresh) |

`Longpipe` is the cost of the effect; compare the other rows with **bypass** on to see whether the
pipeline also changes encode/decode (bigger or busier frames) or starves the encoder.

**Glass to glass (`?flash=1` or the "flash marker" box, applied on Start).** The input is redrawn
into a canvas with a square in its top-left corner that turns white for 100 ms each second, and a
square in the page's top-right corner lights up at the same moment. Film the phone screen with a
second phone in slow motion (240 fps = 4.2 ms per frame) and count frames from the page square
lighting to the corner of the **remote** video lighting: that's everything after capture
(canvas hop ≈ 1 frame, Longpipe, encode, jitter buffer, decode, display). Use background `none`
(or bypass), or the effect may replace the corner. The canvas keeps its starting size, so don't
rotate the phone in this mode.

### Backend option

`EffectsPipeline` takes `backend: 'auto' | 'webgpu' | 'webgl2' | 'none'` (added in 0.0.19 for this;
default `'auto'`). `'webgpu'` still falls back to WebGL2 if WebGPU can't start, `'webgl2'` never tries
WebGPU, and `'none'` skips the GPU entirely (raw passthrough + a non-fatal `no-backend` error), which
is the quickest way to see the fail-open path. Production code should leave it on `'auto'`.

### SDK source vs npm

The example imports `'longpipe'`. Here [`vite.config.js`](vite.config.js) aliases that to this
repo's `src/` (so it always runs the working tree), or to the built `dist/` with
`LONGPIPE=dist npm run example` (after `npm run build`), which is exactly what npm ships.

To use it in your own project: copy `index.html`, `main.js`, `call.js` and `inputs.js`, run
`npm i longpipe vite`, and use a plain Vite config (or none). The alias and the two plugins in
`vite.config.js` exist only to compile Longpipe from source.

## Open it from your phone or another computer

`getUserMedia` only works in a secure context: `https://` or `http://localhost`. Opening
`http://192.168.x.y:5287` from a phone gives "camera not available". Three options:

| Option | How | Tradeoffs |
|---|---|---|
| **LAN + self-signed HTTPS** (simplest) | `npm run example:lan`, then open the `https://192.168.x.y:5287` URL Vite prints | Every device shows a certificate warning once (Safari: "Show Details → visit this website"; Chrome: "Advanced → Proceed"). Same Wi-Fi only. No account, nothing leaves the LAN. |
| **Tunnel** | `npm run example` in one terminal, `cloudflared tunnel --url http://localhost:5287` (or `ngrok http 5287`) in another, then `ALLOWED_HOSTS=<the-tunnel-host> npm run example` | Real HTTPS (no warning), works off-LAN. Video and weights go through a third party; the free URL changes each run. |
| **USB / remote debugging** | Android: `chrome://inspect` → port forwarding `5287 → localhost:5287`, open `http://localhost:5287` on the phone. iOS: no equivalent, use one of the above | Counts as localhost, so no certificate at all. Needs a cable and desktop Chrome. |

On iOS every browser is WebKit, so this is also how to try "Safari" on a phone.

## Manual test checklist (per browser)

Run it on: Chrome (desktop), Firefox (desktop), Safari (macOS), Safari (iOS), Chrome (Android).
Note the browser + version, and the **backend** and **paths** rows, for each.

1. **Start (camera + mic, background blur, denoise on, backend auto).** Allow camera/mic. Within a
   few seconds: status `running`, backend `webgpu` or `webgl2`, the remote video is blurred,
   frames decoded keeps climbing, remote fps is close to output fps. The local video should show
   the raw camera briefly before the effect (that's the passthrough-until-ready behaviour).
2. **Backgrounds.** Switch to solid colour, image, none, blur. Each change is live (no black frame,
   no stall in frames decoded); the colour picker updates the colour live.
3. **Audio.** Tick "hear remote audio" (use headphones, or you'll get feedback). Talk with a fan or
   keyboard noise in the background; toggle denoise on/off and listen for the noise coming back.
   The denoise row should say `rnnoise`/`dfn`/`dfnint8`, `active`. **Safari especially:** confirm
   the remote audio really is denoised (the headless WebKit used by the e2e tests sends the
   un-denoised audio over WebRTC; see `tests/e2e/README.md`).
4. **Backend: force WebGL2.** The pipeline restarts (a short raw-video blip); backend row says
   `webgl2`; effect works. Then **prefer WebGPU**: `webgpu` where the browser has it, else `webgl2`.
5. **Backend: none.** Remote video keeps flowing, raw (no effect); last error shows
   `no-backend` (not fatal); status `passthrough`.
6. **Recovery (optional, from the devtools console):**
   `example.pipeline._debug('lose-backend')`. Last error: `backend-lost`; video keeps flowing raw
   for a moment, then the effect comes back (the log shows `onRecovered`). On Safari the log also
   shows `onTrackChanged … replaced the sender's video track`, and the remote video must keep
   moving afterwards: that's the `replaceTrack` path.
7. **Stop.** Camera light goes off; no new errors in the console.
8. **Phone only:** lock the screen for 10 s and come back; video and audio should resume (if audio
   is silent, tick "hear remote audio" again: that's a user gesture that resumes the AudioContext).
9. **Model switching.** With `model=auto` the model row says e.g. `small (auto: adaptive) · f16 ·
   webgpu`. Pick `xs`, then `large`: the row follows within a second or two, the remote video
   keeps moving (no black frame; frames decoded keeps climbing) and the effect changes quality.
   Pick `MediaPipe selfie`: a short raw blip (new pipeline), then `mediapipe-selfie-landscape`
   (or `-square` in portrait). Pick `medium`: another rebuild back to Longpipe. Note the
   `Longpipe` latency and output fps for each; `xl` is expected to be too slow on phones.
10. **Bypass baseline.** Tick "bypass Longpipe": the remote video turns raw (no effect) within a
    frame or two, status says `bypassed`, `Longpipe` says `bypassed`, frames decoded keeps
    climbing. Write down encode / jitter buffer / decode / estimated total, untick, and compare
    after ~5 s with the pipeline back. The difference in the estimated total is what Longpipe
    adds to the call. Repeat with the phone warm (after a few minutes) if numbers drift.
11. **Glass to glass (optional).** `?flash=1&bg=none`, then film as described in "Reading the
    latency numbers" and compare with the estimated total (expect it ~1-2 frames higher: canvas
    hop + display).
12. **Phone only: rotation.** Portrait: the remote video is upright and portrait (e.g. 720×1280),
   like the local preview. Turn the phone to landscape both ways (left and right) mid-call: the
   remote video turns landscape and stays upright within about a second, with no black frames or
   errors. Repeat with the back camera (`?camera=back`; `?camera=front` is the default on phones).
   On the front camera, hold up some text: the remote video should be mirrored the same way as
   the local preview. iOS Safari doesn't expose the camera's rotation (`VideoFrame.rotation`), so
   the SDK measures it once a second by comparing the `<video>` with the raw frame; with
   `debug: true` the console logs each `orientation … → …` change.
