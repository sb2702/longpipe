// LiveKit smoke test: examples/livekit against a local livekit-server.
// Separate from tests/e2e so the main CI doesn't need a LiveKit server.
//
//   npm run test:e2e:livekit                         Chromium + Firefox
//   LIVEKIT_SERVER=/path/to/livekit-server npm run test:e2e:livekit
//
// Starts `livekit-server --dev` (examples/livekit/.server/, see get-livekit-server.sh)
// unless one already answers on :7880.
import { defineConfig, devices } from '@playwright/test'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const PORT = Number(process.env.E2E_LIVEKIT_PORT ?? 5291)
const SERVER = process.env.LIVEKIT_SERVER ?? fileURLToPath(new URL('../../examples/livekit/.server/livekit-server', import.meta.url))

export default defineConfig({
  testDir: '.',
  testMatch: /.*\.e2e\.mjs$/,
  outputDir: 'results/artifacts',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  workers: 2,
  reporter: [['list']],
  use: { baseURL: `http://localhost:${PORT}`, trace: 'retain-on-failure' },
  webServer: [
    {
      command: existsSync(SERVER) ? `${SERVER} --dev` : 'echo "no livekit-server: run examples/livekit/get-livekit-server.sh" && exit 1',
      url: 'http://localhost:7880/',
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command: `node tests/e2e/fetch_assets.mjs && npx vite --config examples/livekit/vite.config.js --port ${PORT}`,
      cwd: '../..',
      url: `http://localhost:${PORT}/`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required',
          '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] },
      },
    },
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            'webgl.force-enabled': true,
            'media.autoplay.default': 0,
            'media.autoplay.block-webaudio': false,
            'media.navigator.streams.fake': true,
            'media.navigator.permission.disabled': true,
            'media.peerconnection.ice.obfuscate_host_addresses': false,
            'media.peerconnection.ice.loopback': true,
            'media.peerconnection.ice.link_local': true,
          },
        },
      },
    },
  ],
})
