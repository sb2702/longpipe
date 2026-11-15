import { defineConfig } from 'vite'

export default defineConfig({
  root: 'web',
  server: {
    // To open the page from another device through a tunnel (ngrok, cloudflared…):
    //   ALLOWED_HOSTS=your-tunnel.example.com npm run dev
    allowedHosts: process.env.ALLOWED_HOSTS?.split(',') ?? [],
    // Same headers as the article harness this is based on (cross-origin isolation
    // makes SharedArrayBuffer available to WASM runtimes that can use threads).
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
})
