import { defineConfig } from 'vitest/config'
import { resolve } from 'path'
import type { Plugin } from 'vite'

const wgslPlugin: Plugin = {
  name: 'wgsl-text',
  transform(src, id) {
    if (id.endsWith('.wgsl')) return `export default ${JSON.stringify(src)}`
  },
}

export default defineConfig({
  plugins: [wgslPlugin],
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
