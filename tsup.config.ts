import { defineConfig } from 'tsup'
import { resolve } from 'path'
import * as esbuild from 'esbuild'

// Resolves `~/...` imports to absolute paths under src/. Shared with both
// the main build and the inline-worker plugin's nested build below, so the
// alias works inside the worker bundle too.
const tildeAlias: esbuild.Plugin = {
  name: 'tilde-alias',
  // All `~/...` imports include explicit extensions (`.ts`, `.wgsl`, `.glsl`),
  // so this plugin just rewrites the prefix and hands the fully-qualified
  // path to esbuild. No extension/index resolution needed — that lives at
  // the import site.
  setup(build) {
    build.onResolve({ filter: /^~\// }, args => ({
      path: resolve(__dirname, 'src', args.path.slice(2)),
    }))
  },
}

// Bundles the worker entry as a self-contained ESM string and injects it as
// the WORKER_SOURCE export of `src/pipeline/worker_inline.ts`. At runtime
// the pipeline wraps the string in a Blob URL and spawns the worker from
// that — same-origin for any consumer (npm + bundler, esm.sh, jsdelivr,
// etc), avoiding the cross-origin worker / missing-worker-file failure
// modes you get from `new Worker(new URL('./worker/index.ts', import.meta.url))`
// after publishing.
const inlineWorker: esbuild.Plugin = {
  name: 'inline-worker',
  setup(build) {
    build.onLoad({ filter: /pipeline\/worker_inline\.ts$/ }, async () => {
      const result = await esbuild.build({
        entryPoints: [resolve(__dirname, 'src/pipeline/worker/index.ts')],
        bundle: true,
        format: 'esm',
        write: false,
        target: 'es2020',
        minify: true,
        loader: { '.wgsl': 'text', '.glsl': 'text' },
        plugins: [tildeAlias],
      })
      const workerSrc = result.outputFiles[0].text
      return {
        contents: `export const WORKER_SOURCE = ${JSON.stringify(workerSrc)};`,
        loader: 'ts',
      }
    })
  },
}

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  minify: true,
  // Treeshake unused exports — `~/...` paths are fully resolved so esbuild
  // can statically analyze the graph. Trims dead branches in shipped builds.
  treeshake: true,
  loader: {
    '.wgsl': 'text',
    '.glsl': 'text',
  },
  esbuildPlugins: [tildeAlias, inlineWorker],
})
