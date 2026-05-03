import { defineConfig } from 'tsup'
import { resolve } from 'path'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  loader: {
    '.wgsl': 'text',
    '.glsl': 'text',
  },
  esbuildPlugins: [
    {
      name: 'tilde-alias',
      setup(build) {
        build.onResolve({ filter: /^~\// }, args => ({
          path: resolve(__dirname, 'src', args.path.slice(2)),
        }))
      },
    },
  ],
})
