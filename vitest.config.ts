import { defineConfig } from 'vitest/config'
import { resolve } from 'path'
import type { Plugin } from 'vite'

const shaderPlugin: Plugin = {
  name: 'shader-text',
  transform(src, id) {
    if (id.endsWith('.wgsl') || id.endsWith('.glsl') || id.endsWith('.vert') || id.endsWith('.frag'))
      return `export default ${JSON.stringify(src)}`
  },
}

export default defineConfig({
  plugins: [shaderPlugin],
  resolve: {
    alias: {
      '~': resolve(__dirname, 'src'),
    },
  },
  test: {
    browser: {
      enabled: true,
      provider: 'playwright',
      instances: [{ browser: 'chromium' }],
    },
  },
})
