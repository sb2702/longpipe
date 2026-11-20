# LiveKit vs Longpipe, side by side

One camera, two `LocalVideoTrack`s (no Room or server needed): left `@livekit/track-processors`
(MediaPipe selfie, `maskBlending: 'alpha'`, `temporalSmoothing: true`; the label shows the installed version,
and 0.8.3 predates those two options, so they apply only once LiveKit publishes them), right `@longpipe/livekit`, preset `xl`
(suits an M-series Mac; `?preset=large` etc. on slower machines).

**Run on the Mac:** `git pull && npm install && npm run demo:compare`, open http://localhost:5291.
Or run it on the Linux box with `ngrok http 5291 --url katana.ngrok.dev` and open https://katana.ngrok.dev
on the Mac (use `npm run demo:compare -- --mode tunnel` for live reload).

**Record:** add `?clean` (controls hidden, `c` toggles them). Keys: `1` blur, `2` background (again: next
image), `3` none, `t` touch-up, `r` reframe. For a 20 s clip: `1` → `2` → `t` → `r`. Light your face well,
and move your hands near your hair and the frame edges: that's where the mattes differ.
