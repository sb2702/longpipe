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
- **Live readout**: backend actually used, input and output fps, frames sent/decoded and remote
  fps from `getStats()`, received audio, denoise state, transport paths, restarts, last error.
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
npm run example      # http://localhost:5180
```

Click **Start**. "camera + mic" needs a camera; "test clip + tone/noise" doesn't (it loops a
clip of a person into a canvas and synthesises a 440 Hz tone plus noise, which the denoiser should
almost entirely remove: there's no speech in it).

URL parameters preset the controls, which is handy on a phone:
`?input=clip&bg=color&color=%2300ff00&denoise=on&backend=webgl2`. Three more have no control:
`preset=fast|balanced|quality` (default `auto`), `weights=local` (load weights from
`./assets/models/` instead of the CDN) and `debug=1` (verbose SDK logging).

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
`http://192.168.x.y:5180` from a phone gives "camera not available". Three options:

| Option | How | Tradeoffs |
|---|---|---|
| **LAN + self-signed HTTPS** (simplest) | `npm run example:lan`, then open the `https://192.168.x.y:5180` URL Vite prints | Every device shows a certificate warning once (Safari: "Show Details → visit this website"; Chrome: "Advanced → Proceed"). Same Wi-Fi only. No account, nothing leaves the LAN. |
| **Tunnel** | `npm run example` in one terminal, `cloudflared tunnel --url http://localhost:5180` (or `ngrok http 5180`) in another, then `ALLOWED_HOSTS=<the-tunnel-host> npm run example` | Real HTTPS (no warning), works off-LAN. Video and weights go through a third party; the free URL changes each run. |
| **USB / remote debugging** | Android: `chrome://inspect` → port forwarding `5180 → localhost:5180`, open `http://localhost:5180` on the phone. iOS: no equivalent, use one of the above | Counts as localhost, so no certificate at all. Needs a cable and desktop Chrome. |

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
