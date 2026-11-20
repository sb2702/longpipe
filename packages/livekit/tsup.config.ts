import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  dts: { resolve: false },
  external: ['longpipe', 'longpipe/audio', 'livekit-client'],
})
