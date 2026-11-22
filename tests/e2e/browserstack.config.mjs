// The e2e suite on BrowserStack real devices and desktop browsers.
//
//   export BROWSERSTACK_USERNAME=… BROWSERSTACK_ACCESS_KEY=…
//   npm run test:e2e:bs                              all platforms below
//   npm run test:e2e:bs -- --project=ios-safari      one
//
// The built example is served locally over HTTPS (vite preview --mode lan, throwaway
// cert: secure context for WebGPU / AudioWorklet / camera) and reached through
// BrowserStack Local (started by browserstack.global.mjs). Devices see
// localhost as bs-local.com. Each worker (a platform × spec file) is one BrowserStack session.
//
// Only a subset of the suite runs here (GPU-dependent scenarios with 'auto',
// no-GPU, GPU loss, the camera-restart soak) to keep sessions and minutes low.
// The real iPhone camera (a permission prompt Playwright can't answer) is
// covered by tests/e2e/browserstack/ios-camera.mjs.
import { defineConfig } from '@playwright/test'
import base from './playwright.config.mjs'

const PORT = Number(process.env.E2E_PORT ?? 5182)
process.env.E2E_BROWSERSTACK = '1'   // read by the specs (workers load this config too)

// name → BrowserStack capabilities. Playwright engine: chromium (Chrome/Edge),
// firefox / webkit (Playwright's builds on BrowserStack's desktops), webkit for
// real iOS Safari, android (Chrome on a real Android device).
const PLATFORMS = {
  'ios-safari':      { engine: 'webkit', caps: { browser: 'safari', deviceName: 'iPhone Air', osVersion: '26', realMobile: 'true' } },
  'android-chrome':  { engine: 'android', caps: { browser: 'chrome', deviceName: 'Google Pixel 9', osVersion: '15.0', realMobile: 'true' } },
  'macos-webkit':    { engine: 'webkit', caps: { browser: 'playwright-webkit', os: 'OS X', os_version: 'Tahoe' } },
  'windows-chrome':  { engine: 'chromium', caps: { browser: 'chrome', browser_version: 'latest', os: 'Windows', os_version: '11' } },
  'windows-edge':    { engine: 'chromium', caps: { browser: 'edge', browser_version: 'latest', os: 'Windows', os_version: '11' } },
  'windows-firefox': { engine: 'firefox', caps: { browser: 'playwright-firefox', os: 'Windows', os_version: '11' } },
}

export default defineConfig({
  ...base,
  // The subset (see above). Titles end in [backend].
  grep: [/background only \[auto\]/, /background \+ audio \[auto\]/, /GPU device lost mid-call \[auto\]/, /no GPU \(backend none\)/, /camera restart soak/],
  timeout: 240_000,
  retries: 0,
  workers: Number(process.env.E2E_WORKERS ?? 5),   // the OSS plan allows 5 parallel sessions
  fullyParallel: false,                           // one session per worker: keep a platform's tests on it
  outputDir: 'results-bs/artifacts',
  reporter: [
    ['list'],
    ['html', { outputFolder: 'results-bs/html', open: 'never' }],
    ['json', { outputFile: 'results-bs/results.json' }],
    ['./summary-reporter.mjs', { outputDir: 'results-bs' }],
  ],
  globalSetup: './browserstack.global.mjs',
  use: {
    baseURL: `https://localhost:${PORT}`,
    ignoreHTTPSErrors: true,
    trace: 'off',            // not available on real devices; keeps remote calls down elsewhere
    screenshot: 'off',
    video: 'off',            // BrowserStack records every session (dashboard)
  },
  webServer: {
    // The BUILT example (npm run build + vite build, LONGPIPE=dist), served by vite
    // preview over HTTPS: the dev server's hundreds of unbundled modules don't load
    // reliably through the tunnel on iOS (the page never ran), and the build is what
    // `npm i longpipe` users ship anyway.
    command: 'npm run e2e:bs:serve',
    cwd: '../..',
    url: `https://localhost:${PORT}/`,
    ignoreHTTPSErrors: true,
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
    env: { E2E_PORT: String(PORT) },
  },
  projects: Object.entries(PLATFORMS).map(([name, p]) => ({
    name,
    use: {
      browserName: p.engine === 'android' ? 'chromium' : p.engine,
      // Devices and macOS reach the tunnel at bs-local.com (localhost isn't remapped there).
      ...(p.caps.realMobile || p.caps.os === 'OS X' ? { baseURL: `https://bs-local.com:${PORT}` } : {}),
    },
    metadata: { browserstack: { caps: p.caps, android: p.engine === 'android' } },
  })),
})
