// End-to-end smoke tests: the vanilla-WebRTC example (examples/vanilla-webrtc)
// in real Chromium, Firefox and WebKit, through a real RTCPeerConnection.
//
//   npm run test:e2e                       all three browsers
//   npm run test:e2e -- --project=firefox  one browser
//
// Results: tests/e2e/results/summary.md (matrix), results/html (Playwright
// report), results/results.json. See tests/e2e/README.md.
import { defineConfig, devices } from '@playwright/test'

const PORT = Number(process.env.E2E_PORT ?? 5181)

// Headless Linux Chromium has no WebGPU adapter by default.
// --enable-unsafe-webgpu exposes SwiftShader's (CPU) WebGPU adapter; WebGL2 is
// SwiftShader too (--enable-unsafe-swiftshader keeps that fallback allowed).
// E2E_CHROMIUM_GPU=vulkan uses the host GPU through Vulkan instead (local runs
// on a machine with a GPU + Mesa/Vulkan driver; not on CI).
const chromiumGpuArgs = process.env.E2E_CHROMIUM_GPU === 'vulkan'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist']
  : ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader']

export default defineConfig({
  testDir: '.',
  testMatch: /.*\.e2e\.mjs$/,
  outputDir: 'results/artifacts',
  timeout: 120_000,
  expect: { timeout: 20_000 },
  // Software GPUs are slow: two pipelines at once is the most that stays
  // representative on a 2-4 core CI runner.
  workers: Number(process.env.E2E_WORKERS ?? 2),
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'results/html', open: 'never' }],
    ['json', { outputFile: 'results/results.json' }],
    ['./summary-reporter.mjs', { outputDir: 'results' }],
  ],
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    // Fetches the pinned clip + weights (no-op when present), then serves the
    // example compiled from this repo's src/ (LONGPIPE=dist for the build).
    command: `node tests/e2e/fetch_assets.mjs && npx vite --config examples/vanilla-webrtc/vite.config.js --port ${PORT}`,
    cwd: '../..',
    url: `http://localhost:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            ...chromiumGpuArgs,
            '--autoplay-policy=no-user-gesture-required',
            '--use-fake-ui-for-media-stream',
            '--use-fake-device-for-media-stream',
            // gc(): the unclosed-VideoFrame regression test forces collection.
            '--js-flags=--expose-gc',
          ],
        },
      },
    },
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        launchOptions: {
          firefoxUserPrefs: {
            // CI runners have no GPU: allow WebGL on Mesa's software rasteriser.
            'webgl.force-enabled': true,
            'media.autoplay.default': 0,
            'media.autoplay.block-webaudio': false,
            'media.navigator.streams.fake': true,
            'media.navigator.permission.disabled': true,
            // Loopback ICE without mDNS host candidates (no mDNS resolver on CI).
            'media.peerconnection.ice.obfuscate_host_addresses': false,
            // CI runners: allow loopback/link-local ICE candidates so the in-page loopback call connects.
            'media.peerconnection.ice.loopback': true,
            'media.peerconnection.ice.link_local': true,
          },
        },
      },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
})
