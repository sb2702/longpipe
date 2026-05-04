import { defineConfig } from 'vite'
import { resolve } from 'node:path'
import type { Plugin } from 'vite'

const shaderPlugin: Plugin = {
  name: 'shader-text',
  transform(src, id) {
    if (id.endsWith('.wgsl') || id.endsWith('.glsl') || id.endsWith('.vert') || id.endsWith('.frag'))
      return `export default ${JSON.stringify(src)}`
  },
}

// Demo lives at sdk/demo. Sources at sdk/src (~). Generated weights at
// sdk/weights — served via fs.allow so the demo can fetch /weights/...
export default defineConfig({
  root: __dirname,
  plugins: [shaderPlugin],
  resolve: {
    alias: {
      '~': resolve(__dirname, '../src'),
    },
  },
  server: {
    fs: { allow: [resolve(__dirname, '..')] },
  },
})
