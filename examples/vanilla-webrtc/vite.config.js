// Vite config for the vanilla-WebRTC example.
//
//   npm run example        http://localhost:5287 (getUserMedia works on localhost)
//   npm run example:lan    https://<this machine's LAN IP>:5287, self-signed cert
//                          (open it from a phone / another computer; see README)
//
// By default `import ... from 'longpipe'` is compiled from this repo's src/,
// so the example (and the e2e tests that drive it) exercise the working tree.
// LONGPIPE=dist uses the built package in ../../dist instead (run `npm run
// build` first): that's exactly what `npm i longpipe` ships.
//
// In your own app you need none of this: `npm i longpipe`, delete the alias and
// the plugins below, and Vite (or any bundler) resolves 'longpipe' from
// node_modules. The published build inlines its worker and audio worklet.
import { defineConfig } from 'vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import * as esbuild from 'esbuild'

const REPO = fileURLToPath(new URL('../../', import.meta.url))
const SRC = resolve(REPO, 'src')
const useDist = process.env.LONGPIPE === 'dist'

// ── Only needed to compile Longpipe from source (same as demo/vite.config.ts) ──
// Shaders are imported as text.
const shaderText = {
  name: 'longpipe-shader-text',
  transform(src, id) {
    if (/\.(wgsl|glsl|vert|frag)$/.test(id)) return `export default ${JSON.stringify(src)}`
  },
}
// AudioWorklet scopes can't import modules, so the denoise processor is
// pre-bundled into a string (tsup does this for the published build).
const tildeAlias = {
  name: 'tilde-alias',
  setup(build) {
    build.onResolve({ filter: /^~\// }, (args) => ({ path: resolve(SRC, args.path.slice(2)) }))
  },
}
const inlineWorklet = {
  name: 'longpipe-inline-worklet',
  async load(id) {
    if (!id.replace(/\\/g, '/').split('?')[0].endsWith('src/audio/worklet_inline.ts')) return
    const out = await esbuild.build({
      entryPoints: [resolve(SRC, 'audio/worklet/processor.ts')],
      bundle: true, format: 'esm', write: false, target: 'es2020', minify: false, plugins: [tildeAlias],
    })
    return `export const WORKLET_SOURCE = ${JSON.stringify(out.outputFiles[0].text)};`
  },
}

export default defineConfig(({ mode }) => ({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [
    ...(useDist ? [] : [shaderText, inlineWorklet]),
    // `--mode lan`: serve HTTPS with a throwaway self-signed certificate, so
    // getUserMedia works when the page is opened by IP from another device.
    ...(mode === 'lan' ? [basicSsl()] : []),
  ],
  resolve: {
    alias: useDist
      ? [{ find: /^longpipe$/, replacement: resolve(REPO, 'dist/index.js') }]
      : [
          { find: /^longpipe$/, replacement: resolve(SRC, 'index.ts') },
          { find: /^~\//, replacement: SRC + '/' },
        ],
  },
  server: {
    port: Number(process.env.PORT ?? 5287),
    strictPort: true,
    host: mode === 'lan' ? true : 'localhost',
    // Tunnels (ngrok, cloudflared) give real HTTPS, which iOS Safari needs for the camera.
    // Common tunnel domains are allowed; add others with ALLOWED_HOSTS=host1,host2.
    allowedHosts: ['.ngrok.dev', '.ngrok.app', '.ngrok-free.app', '.trycloudflare.com',
      ...(process.env.ALLOWED_HOSTS?.split(',') ?? [])],
    // `--mode tunnel`: the page is served over the tunnel's HTTPS on 443, so point
    // Vite's live-reload websocket there too (otherwise it can't connect and keeps polling).
    ...(mode === 'tunnel' ? { hmr: { clientPort: 443, protocol: 'wss' } } : {}),
    fs: { allow: [REPO] },
  },
  // Longpipe ships its own worker; don't let the dep optimizer touch it.
  optimizeDeps: { exclude: ['longpipe'] },
}))
