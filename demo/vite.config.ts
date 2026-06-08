import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import { readFile } from 'node:fs/promises'
import * as esbuild from 'esbuild'
import type { Plugin } from 'vite'

const shaderPlugin: Plugin = {
  name: 'shader-text',
  transform(src, id) {
    if (id.endsWith('.wgsl') || id.endsWith('.glsl') || id.endsWith('.vert') || id.endsWith('.frag'))
      return `export default ${JSON.stringify(src)}`
  },
}

// AudioWorklet scopes can't ESM-import, so the processor must be pre-bundled.
// tsup does this for published builds (WORKLET_SOURCE); in dev we do the same
// on the fly — bundle src/audio/worklet/processor.ts and inject it when vite
// loads worklet_inline.ts. Mirrors the inline-worker/worklet tsup plugins.
const tildeAlias: esbuild.Plugin = {
  name: 'tilde-alias',
  setup(build) {
    build.onResolve({ filter: /^~\// }, (args) => ({ path: resolve(__dirname, '../src', args.path.slice(2)) }))
  },
}
const inlineWorkletPlugin: Plugin = {
  name: 'inline-worklet-dev',
  async load(id) {
    const clean = id.replace(/\\/g, '/').split('?')[0]
    if (!clean.endsWith('src/audio/worklet_inline.ts')) return
    const out = await esbuild.build({
      entryPoints: [resolve(__dirname, '../src/audio/worklet/processor.ts')],
      // minify OFF in dev so runtime errors carry real names + readable stacks
      // (tsup keeps minify on for published builds).
      bundle: true, format: 'esm', write: false, target: 'es2020', minify: false,
      plugins: [tildeAlias],
    })
    return `export const WORKLET_SOURCE = ${JSON.stringify(out.outputFiles[0].text)};`
  },
}

// Serve the staged weights dir at /weights/ so a demo can load assets locally
// (model_*.bin + the audio dfn.wasm / packs) before they're uploaded to the CDN.
const CT: Record<string, string> = {
  '.wasm': 'application/wasm', '.bin': 'application/octet-stream',
  '.pack': 'application/octet-stream', '.json': 'application/json',
}
const serveWeights: Plugin = {
  name: 'serve-weights',
  configureServer(server) {
    server.middlewares.use('/weights', async (req, res, next) => {
      try {
        const rel = decodeURIComponent((req.url ?? '/').split('?')[0])
        const file = resolve(__dirname, '../weights', '.' + rel)
        const body = await readFile(file)
        res.setHeader('content-type', CT[file.slice(file.lastIndexOf('.'))] ?? 'application/octet-stream')
        res.end(body)
      } catch { next() }
    })
  },
}

// Demo lives at sdk/demo. Sources at sdk/src (~). Weights at sdk/weights.
export default defineConfig({
  root: __dirname,
  plugins: [shaderPlugin, inlineWorkletPlugin, serveWeights],
  resolve: {
    alias: {
      '~': resolve(__dirname, '../src'),
    },
  },
  server: {
    host: true,           // bind 0.0.0.0 — reachable from LAN / containers
    allowedHosts: true,   // accept any Host header (vite 5+ strict-checks otherwise)
    fs: { allow: [resolve(__dirname, '..')] },
  },
})
