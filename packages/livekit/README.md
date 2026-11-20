# @longpipe/livekit

[Longpipe](https://longpipe.dev) video effects as LiveKit track processors: background blur, virtual backgrounds, touch-up, auto-reframe, and speech denoise. WebGPU with a WebGL2 fallback; Chromium, Firefox and Safari.

```
npm install @longpipe/livekit longpipe livekit-client
```

```ts
import { BackgroundBlur, LongpipeProcessor, DenoiseProcessor, EffectsPipeline } from '@longpipe/livekit'

EffectsPipeline.unlockAudio()                       // in the join click, before any await (denoise)
const processor = BackgroundBlur()                  // same names as @livekit/track-processors
await cameraTrack.setProcessor(processor)
await processor.switchTo({ mode: 'virtual-background', imagePath: '/beach.jpg' })

// Longpipe extras: any EffectsPipeline option
await cameraTrack.setProcessor(LongpipeProcessor({ background: 'blur', touchup: true, reframe: true }))

// Denoise: set it after the mic track is published (LiveKit sets the AudioContext on publish)
await micTrack.setProcessor(DenoiseProcessor())
```

`BackgroundBlur(radius?)`, `VirtualBackground(url)` and `BackgroundProcessor({ mode })` mirror `@livekit/track-processors`, so switching is an import change. `processor.pipeline` is the underlying [`EffectsPipeline`](https://longpipe.dev/docs/api) (`setTouchup`, `getStats`, `ready`…). Camera switches keep the same processed track.

Docs: [longpipe.dev/docs/livekit](https://longpipe.dev/docs/livekit) · Example: [examples/livekit](https://github.com/sb2702/longpipe/tree/main/examples/livekit) · MIT
