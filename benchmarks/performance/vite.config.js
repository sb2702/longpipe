import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

// The "as shipped" pipelines drive the SDK's own per-frame renderer
// (src/pipeline/worker/renderer.ts), which `longpipe` does not export: it is
// compiled from this repo's src/ (the same `~/` alias and shader-as-text loading
// as the SDK's tsup build and demo/vite.config.ts).
const SRC = fileURLToPath(new URL('../../src/', import.meta.url))
const REPO = fileURLToPath(new URL('../../', import.meta.url))

const shaderText = {
  name: 'shader-text',
  transform(src, id) {
    if (/\.(wgsl|glsl)$/.test(id)) return `export default ${JSON.stringify(src)}`
  },
}

export default defineConfig({
  root: 'web',
  plugins: [shaderText],
  resolve: { alias: [{ find: /^~\//, replacement: SRC }] },
  server: {
    // To open the page from another device through a tunnel (ngrok, cloudflared…):
    //   ALLOWED_HOSTS=your-tunnel.example.com npm run dev
    allowedHosts: process.env.ALLOWED_HOSTS?.split(',') ?? [],
    // Same headers as the article harness this is based on (cross-origin isolation
    // makes SharedArrayBuffer available to WASM runtimes that can use threads).
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
    fs: { allow: [REPO] },
  },
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision', '@litertjs/core'] },
})
