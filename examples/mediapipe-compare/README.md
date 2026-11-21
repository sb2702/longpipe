# MediaPipe vs Longpipe, side by side

One camera, no WebRTC. Left: MediaPipe's `selfie_segmenter` (the model behind Jitsi's and LiveKit's background
effects), GPU delegate, its confidence mask used directly as alpha, no smoothing over time. Right: Longpipe
`EffectsPipeline`, preset `xl` (`?preset=large` etc. on slower machines). Both draw the same blur or image.

**Run:** `npm install && npm run demo:mediapipe`, open http://localhost:5292.
**Tunnel:** `npm run demo:mediapipe:tunnel`, then `ngrok http 5292 --url katana.ngrok.dev`, open https://katana.ngrok.dev.

**Record:** add `?clean`. Keys: `1` blur, `2` background (again: next image), `3` none, `t` touch-up,
`r` reframe, `c` controls. Move your hands near your hair and the frame edges: that's where the mattes differ.
