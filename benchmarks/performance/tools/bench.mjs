// Runs web/index.html in your local Google Chrome (via Playwright), prints the
// results and saves one JSON report (host, library versions, config,
// environment, every run) to results/.
//
//   npm run bench -- [--samples 360p,720p] [--modes transcode,process,model] [--runs 5]
//                    [--pipelines passthrough,longpipe-webgpu-large,...]
//                    [--dtype f16|f32] [--headless] [--check] [--out results/my-machine.json]
//
// Adapted from github.com/sb2702/webrtc-hacks-webgpu (segmentation/tools/bench.mjs, MIT).
import { execSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import { parseArgs } from 'node:util'
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

const { values: args } = parseArgs({
  options: {
    samples: { type: 'string', default: '360p,720p' },
    modes: { type: 'string', default: 'transcode,process,model' },
    runs: { type: 'string', default: '5' },
    pipelines: { type: 'string' },
    dtype: { type: 'string' },
    headless: { type: 'boolean' },
    check: { type: 'boolean' },
    thumbs: { type: 'string' },
    out: { type: 'string' },
  },
})

const root = new URL('../', import.meta.url).pathname
const version = (pkg) => JSON.parse(readFileSync(`${root}node_modules/${pkg}/package.json`, 'utf8')).version
const sh = (cmd) => { try { return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim() } catch { return null } }

// Linux needs Vulkan for real-GPU WebGPU (otherwise SwiftShader, a CPU emulation).
const chromeArgs = process.platform === 'linux'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist', ...(args.headless ? ['--use-angle=vulkan'] : [])]
  : ['--enable-unsafe-webgpu']

const server = await createServer({ configFile: `${root}vite.config.js`, server: { port: 0 }, logLevel: 'error' })
await server.listen()
const origin = server.resolvedUrls.local[0].replace(/\/$/, '')
const browser = await chromium.launch({
  ...(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: 'chrome' }),
  headless: !!args.headless,
  args: [...chromeArgs, ...(args.headless ? [] : ['--window-size=1000,800'])],
})
const page = await browser.newPage()
const NOISE = /404|GPU stall due to ReadPixels|is intended for demos|^(INFO|WARNING): |^[IW]\d{4} |=== Source Location/
page.on('console', (m) => ['error', 'warning'].includes(m.type()) && !NOISE.test(m.text()) && console.error(`[page ${m.type()}]`, m.text()))
page.on('pageerror', (e) => console.error('[page]', e.message))
page.setDefaultTimeout(0)

const host = {
  platform: process.platform, release: os.release(), arch: os.arch(), cpu: os.cpus()[0]?.model, cpus: os.cpus().length,
  memoryGB: Math.round(os.totalmem() / 2 ** 30), loadavg: os.loadavg(),
  chrome: browser.version(), headless: !!args.headless,
  ...(process.platform === 'darwin' && {
    macos: sh('sw_vers -productVersion'), model: sh('sysctl -n hw.model'), chip: sh('sysctl -n machdep.cpu.brand_string'),
    power: sh('pmset -g batt'), lowPowerMode: sh("pmset -g | grep -i lowpowermode"),
  }),
}
const versions = Object.fromEntries(['longpipe', '@mediapipe/tasks-vision', 'webcodecs-utils', 'vite', 'playwright-core'].map((p) => [p, version(p)]))
console.log(JSON.stringify({ host, versions }, null, 2))

const reports = []
for (const sample of args.samples.split(',')) {
  const query = new URLSearchParams({ sample, runs: args.runs, ...(args.pipelines && { pipelines: args.pipelines }), ...(args.dtype && { dtype: args.dtype }) })
  if (args.check) {
    await page.goto(`${origin}/?${query}`)
    await page.waitForFunction(() => window.bench)
    console.log(`\n${sample}: ${await page.textContent('#env')}`)
    const checks = await page.evaluate(() => window.bench.checkAll())
    for (const [name, r] of Object.entries(checks)) {
      console.log(name.padEnd(28) + (r.error ? `ERROR ${r.error}` :
        `background ${(r.background * 100).toFixed(1)}%` + (r.agreement != null ? `  agrees with mediapipe-gpu on ${(r.agreement * 100).toFixed(1)}%` : '')))
      if (args.thumbs && r.thumbnail) {
        mkdirSync(args.thumbs, { recursive: true })
        writeFileSync(`${args.thumbs}/${sample}-${name}.jpg`, Buffer.from(r.thumbnail.split(',')[1], 'base64'))
      }
    }
    continue
  }
  for (const mode of args.modes.split(',')) {
    // Fresh page per (sample, mode), so one configuration can't leak GPU state into the next.
    await page.goto(`${origin}/?${new URLSearchParams({ ...Object.fromEntries(query), mode })}`)
    await page.waitForFunction(() => window.bench)
    if (!reports.length) console.log(await page.textContent('#env'))
    const report = await page.evaluate(() => window.bench.runAll())
    reports.push(report)
    const { config: c } = report
    console.log(`\n${c.width}x${c.height}, mode ${c.mode}, ${c.runs} timed runs`)
    for (const r of report.results) {
      if (r.error) { console.log(`${r.name.padEnd(28)}ERROR ${r.error}`); continue }
      const [med, lo, hi] = mode === 'model' ? [r.medianMs, r.minMs, r.maxMs] : [r.medianFps, r.minFps, r.maxFps]
      if (med == null) continue
      console.log(r.name.padEnd(28) + [med, lo, hi].map((f) => f.toFixed(mode === 'model' ? 2 : 1).padStart(9)).join('') +
        `  mask ${r.maskRes ?? '-'}${r.deliveredMaskRes ? ` (delivered ${r.deliveredMaskRes})` : ''}`)
    }
  }
}

if (reports.length) {
  const out = args.out ?? `results/${new Date().toISOString().slice(0, 10)}-${process.platform}-${os.arch()}.json`
  mkdirSync('results', { recursive: true })
  writeFileSync(out, JSON.stringify({ host, versions, reports }, null, 2) + '\n')
  console.log(`\nsaved ${out}`)
}
await browser.close()
await server.close()
