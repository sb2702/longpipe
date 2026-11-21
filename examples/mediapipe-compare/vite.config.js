// Vite config for the MediaPipe-vs-Longpipe page: the vanilla-WebRTC example's config
// (Longpipe from src/, tunnel hosts allowed, `--mode tunnel` for HMR over a tunnel).
// @mediapipe/tasks-vision comes from the root node_modules (a dependency of the LiveKit example).
import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { readFileSync } from 'node:fs'
import vanilla from '../vanilla-webrtc/vite.config.js'

const HERE = fileURLToPath(new URL('.', import.meta.url))
const REPO = resolve(HERE, '../..')
const MP_VERSION = JSON.parse(readFileSync(resolve(REPO, 'node_modules/@mediapipe/tasks-vision/package.json'), 'utf8')).version

export default defineConfig((env) => {
  const base = vanilla(env)
  return {
    ...base,
    root: HERE,
    define: { __MP_VERSION__: JSON.stringify(MP_VERSION) },
    publicDir: resolve(REPO, 'examples/vanilla-webrtc/public'),
    server: { ...base.server, port: Number(process.env.PORT ?? 5292) },
    optimizeDeps: { exclude: ['longpipe'] },
  }
})
