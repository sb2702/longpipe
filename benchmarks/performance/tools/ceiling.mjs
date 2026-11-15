// Output-path investigation driver: runs web/ceiling.html in Chrome for a
// matrix of routes x encoder options x samples and prints/saves the results.
//
//   node tools/ceiling.mjs [--headless] [--samples 360p,720p] [--routes a,b] [--encoders default,realtime]
//                          [--encoder-only] [--chrome-args "--flag1 --flag2"] [--out results/x.json]
import { mkdirSync, writeFileSync } from 'node:fs'
import { parseArgs } from 'node:util'
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

const { values: args } = parseArgs({ options: {
  headless: { type: 'boolean' }, samples: { type: 'string', default: '360p,720p' },
  routes: { type: 'string' }, encoders: { type: 'string', default: 'default' },
  'encoder-only': { type: 'boolean' }, readback: { type: 'boolean' }, 'readback-depth': { type: 'string', default: '0' }, 'chrome-args': { type: 'string', default: '' }, out: { type: 'string' }, runs: { type: 'string', default: '3' },
} })
const root = new URL('../', import.meta.url).pathname
const server = await createServer({ configFile: `${root}vite.config.js`, server: { port: 0 }, logLevel: 'error' })
await server.listen()
const origin = server.resolvedUrls.local[0].replace(/\/$/, '')
const extra = args['chrome-args'].split(' ').filter(Boolean)
const browser = await chromium.launch({
  ...(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: 'chrome' }),
  headless: !!args.headless, args: ['--enable-unsafe-webgpu', ...extra, ...(args.headless ? [] : ['--window-size=1000,800'])],
})
const page = await browser.newPage()
page.on('pageerror', (e) => console.error('[page]', e.message))
page.on('console', (m) => m.type() === 'error' && console.error('[page error]', m.text()))
page.setDefaultTimeout(0)
await page.goto(`${origin}/ceiling.html`)
await page.waitForFunction(() => window.ceiling)
const env = await page.evaluate(() => window.ceiling.environment())
console.log(browser.version(), args.headless ? 'headless' : 'headed', extra.join(' '), env.webgpu?.description ?? env.webgpu?.architecture)
const results = []
if (args.readback) {
  for (const sample of args.samples.split(',')) {
    const [w, h] = sample === '720p' ? [1280, 720] : [640, 360]
    for (const depth of [1, 2, 4, 8]) {
      const r = []
      for (let i = 0; i < 3; i++) r.push(await page.evaluate(([a, b, d]) => window.ceiling.readbackThroughput(a, b, 192, d), [w, h, depth]))
      results.push({ kind: 'readback', sample, depth, runs: r })
      console.log(`${sample} copyTo readback, ${depth} in flight: ${r.map((x) => x.fps.toFixed(0)).join(' / ')} fps`)
    }
  }
  args.samples = ''
}
for (const sample of args.samples.split(',').filter(Boolean)) {
  for (const encoder of args.encoders.split(',')) {
    if (args['encoder-only']) {
      for (const source of ['decoded', 'webgpu-canvas', 'cpu-rgba', 'cpu-nv12']) {
        const r = await page.evaluate(([s, sa, e]) => window.ceiling.encoderOnly(s, sa, e), [source, sample, encoder])
        results.push({ kind: 'encoder-only', sample, encoder, source, runs: r })
        console.log(`${sample} encoder-only ${encoder.padEnd(16)} ${source.padEnd(18)} ${r.map((x) => x.fps.toFixed(0)).join(' / ')} fps`)
      }
      continue
    }
    const routes = args.routes?.split(',') ?? await page.evaluate(() => window.ceiling.routes)
    for (const route of routes) {
      const depth = Number(args['readback-depth'])
      const r = await page.evaluate(([ro, sa, e, n, d]) => window.ceiling.run(ro, sa, e, n, d), [route, sample, encoder, Number(args.runs), depth])
      results.push({ kind: 'transcode', sample, encoder, route, readbackDepth: depth, runs: r })
      console.log(`${sample} ${encoder.padEnd(16)} readback ${depth} ${route.padEnd(20)} ${r.map((x) => x.fps.toFixed(0)).join(' / ')} fps  (in process(): ${r.map((x) => x.medianProcessMs.toFixed(2)).join(' / ')} ms)`)
    }
  }
}
if (args.out) {
  mkdirSync('results', { recursive: true })
  writeFileSync(args.out, JSON.stringify({ chrome: browser.version(), headless: !!args.headless, chromeArgs: extra, environment: env, results }, null, 2) + '\n')
}
await browser.close()
await server.close()
