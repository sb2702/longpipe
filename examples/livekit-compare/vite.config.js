// Vite config for the LiveKit-vs-Longpipe comparison page: the vanilla-WebRTC
// example's config (Longpipe from src/, ngrok/cloudflared hosts allowed,
// `--mode tunnel` for HMR over the tunnel) plus @longpipe/livekit from
// packages/livekit/src. livekit-client and @livekit/track-processors come from
// the root node_modules (installed for the examples/livekit workspace).
import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { readFileSync } from 'node:fs'
import vanilla from '../vanilla-webrtc/vite.config.js'

const HERE = fileURLToPath(new URL('.', import.meta.url))
const REPO = resolve(HERE, '../..')
// Shown on the LiveKit panel label: the installed @livekit/track-processors version.
const LK_TP_VERSION = JSON.parse(readFileSync(resolve(REPO, 'node_modules/@livekit/track-processors/package.json'), 'utf8')).version

export default defineConfig((env) => {
  const base = vanilla(env)
  return {
    ...base,
    root: HERE,
    define: { __LK_TP_VERSION__: JSON.stringify(LK_TP_VERSION) },
    publicDir: resolve(REPO, 'examples/vanilla-webrtc/public'),   // ?input=clip (npm run e2e:assets)
    resolve: {
      alias: [
        { find: /^@longpipe\/livekit$/, replacement: resolve(REPO, 'packages/livekit/src/index.ts') },
        { find: /^longpipe\/audio$/, replacement: resolve(REPO, 'src/audio.ts') },
        ...base.resolve.alias,
      ],
    },
    server: { ...base.server, port: Number(process.env.PORT ?? 5291) },
    optimizeDeps: { exclude: ['longpipe', '@longpipe/livekit'] },
  }
})
