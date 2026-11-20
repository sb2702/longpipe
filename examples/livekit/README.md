# Longpipe + LiveKit

Two participants in one page: **alice** publishes the camera through `@longpipe/livekit`, **bob**
subscribes, so the right-hand video is what LiveKit delivered. Controls: background blur / image /
green / none, denoise on/off, and Longpipe vs LiveKit's own `@livekit/track-processors` for comparison.

The Longpipe code is `applyVideoProcessor()` and `applyDenoise()` in [`main.js`](main.js).

## Run

```bash
npm install                                  # repo root (npm workspaces)
examples/livekit/get-livekit-server.sh       # official release binary → examples/livekit/.server/ (no sudo)
examples/livekit/.server/livekit-server --dev   # ws://localhost:7880, key devkey / secret secret
npm run example:livekit                      # http://localhost:5290
```

Click **Join**. `/token` is a tiny Vite endpoint that mints access tokens with `livekit-server-sdk`
and the dev key. Point it elsewhere with `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.

LiveKit ships no macOS binary: `brew install livekit`, or run the Linux server and forward
ports 7880/7881 (`ssh -R`), and start it with `--node-ip` set to an address the Mac can reach.

No camera? `?input=clip` uses the e2e test clip (`npm run e2e:assets` first).

## Test

```bash
npm run test:e2e:livekit     # Chromium + Firefox; starts livekit-server --dev if none is running
```

Checks what bob receives: the green background in the corners, a source swap (camera switch)
keeping the same processed track, camera mute/unmute (LiveKit stops the camera; the processor
rebuilds), denoise active, and fail-open (`backend: 'none'` still sends raw video).
