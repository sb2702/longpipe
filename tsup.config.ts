import { defineConfig } from 'tsup'
import { resolve } from 'path'

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
  esbuildPlugins: [
    {
      name: 'tilde-alias',
      // All `~/...` imports include explicit extensions (`.ts`, `.wgsl`,
      // `.glsl`), so this plugin just rewrites the prefix and hands the
      // fully-qualified path to esbuild. No extension/index resolution
      // needed — that responsibility lives at the import site.
      setup(build) {
        build.onResolve({ filter: /^~\// }, args => ({
          path: resolve(__dirname, 'src', args.path.slice(2)),
        }))
      },
    },
  ],
})
