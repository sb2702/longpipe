// Runs web/index.html in your local Google Chrome (via Playwright), prints the
// results and saves one JSON report (host, library versions, config,
// environment, every run) to results/.
//
//   npm run bench -- [--samples 360p,720p] [--modes transcode,process,model] [--runs 5]
//                    [--pipelines passthrough,shipped-webgpu-xs,...] [--output readback|direct]
//                    [--busy-cmd 'pgrep -f "[s]ome_gpu_job"'] [--headless] [--check] [--out results/my-machine.json]
//
// Protocol: for each (sample, mode), `--runs` rounds; each round measures every
// pipeline once (create -> warm-up pass -> one timed pass -> destroy), in an
// order rotated by one position per round, on a freshly loaded page. So the
// pipelines are interleaved in time and slow drift (thermals, background load)
// spreads across all of them instead of landing on whichever ran last.
//
// --busy-cmd: a shell command polled once per second; exit status 0 means the
// machine is busy with other work (e.g. a background GPU job). A measurement
// whose timed part overlapped a busy sample (±1.5 s) is discarded and repeated
// once the machine has been idle for 5 s; discarded measurements are kept in the
// report. With `pgrep -f`, write the pattern so it can't match this script's own
// command line, e.g. --busy-cmd 'pgrep -f "[m]y_gpu_job"'.
//
// Adapted from github.com/sb2702/webrtc-hacks-webgpu (segmentation/tools/bench.mjs, MIT).
import { exec, execSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import { parseArgs } from 'node:util'
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

const { values: args } = parseArgs({
  options: {
    samples: { type: 'string', default: '360p,720p' },
    modes: { type: 'string', default: 'transcode' },
    runs: { type: 'string', default: '5' },
    pipelines: { type: 'string' },
    output: { type: 'string', default: 'readback' },
    'busy-cmd': { type: 'string' },
    headless: { type: 'boolean' },
    check: { type: 'boolean' },
    thumbs: { type: 'string' },
    out: { type: 'string' },
  },
})

const root = new URL('../', import.meta.url).pathname
const version = (pkg) => JSON.parse(readFileSync(`${root}node_modules/${pkg}/package.json`, 'utf8')).version
const sh = (cmd) => { try { return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim() } catch { return null } }
const median = (xs) => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2 }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ---- background-load guard --------------------------------------------------
const busy = []   // { t, busy }
let busyTimer
if (args['busy-cmd']) {
  const poll = () => exec(args['busy-cmd'], (err) => busy.push({ t: Date.now(), busy: !err }))
  poll()
  busyTimer = setInterval(poll, 1000)
}
const overlapsBusy = (t0, t1) => busy.some((s) => s.busy && s.t >= t0 - 1500 && s.t <= t1 + 1500)
async function waitIdle() {
  if (!args['busy-cmd']) return
  let announced = false
  for (;;) {
    const recent = busy.filter((s) => s.t > Date.now() - 5000)
    if (recent.length >= 4 && recent.every((s) => !s.busy)) return
    if (!announced && recent.some((s) => s.busy)) { console.log('  (machine busy: waiting for 5 s idle)'); announced = true }
    await sleep(1000)
  }
}

// ---- browser ----------------------------------------------------------------
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
const load = async (query = {}) => {
  await page.goto(`${origin}/?${new URLSearchParams(query)}`)
  await page.waitForFunction(() => window.bench)
}

const host = {
  platform: process.platform, release: os.release(), arch: os.arch(), cpu: os.cpus()[0]?.model, cpus: os.cpus().length,
  memoryGB: Math.round(os.totalmem() / 2 ** 30), loadavgAtStart: os.loadavg(),
  chrome: browser.version(), headless: !!args.headless,
  ...(process.platform === 'darwin' && {
    macos: sh('sw_vers -productVersion'), model: sh('sysctl -n hw.model'), chip: sh('sysctl -n machdep.cpu.brand_string'),
    power: sh('pmset -g batt'), lowPowerMode: sh("pmset -g | grep -i lowpowermode"), thermalAtStart: sh('pmset -g therm'),
  }),
}
const versions = Object.fromEntries(['longpipe', '@mediapipe/tasks-vision', '@litertjs/core', 'webcodecs-utils', 'vite', 'playwright-core'].map((p) => [p, version(p)]))
// Row 5 (shipped-*) is compiled from this repo's src/: record exactly which source.
const sdkSource = {
  commit: sh(`git -C ${root} rev-parse HEAD`),
  srcClean: sh(`git -C ${root} status --porcelain -- ../../src`) === '',
  srcVersion: JSON.parse(readFileSync(`${root}../../package.json`, 'utf8')).version,
}
console.log(JSON.stringify({ host, versions, sdkSource }, null, 2))

await load()
const allNames = await page.evaluate(() => window.bench.pipelines())
const names = args.pipelines ? args.pipelines.split(',') : allNames
for (const n of names) if (!allNames.includes(n)) throw new Error(`unknown pipeline ${n}`)

// ---- --check: one frame per pipeline, background share + agreement --------
if (args.check) {
  for (const sample of args.samples.split(',')) {
    await load({ sample, pipelines: names.join(',') })
    console.log(`\n${sample}: ${await page.textContent('#env')}`)
    const checks = await page.evaluate(() => window.bench.checkAll())
    for (const [name, r] of Object.entries(checks)) {
      console.log(name.padEnd(30) + (r.error ? `ERROR ${r.error}` :
        `background ${(r.background * 100).toFixed(1)}%` + (r.agreement != null ? `  agrees with mediapipe-gpu on ${(r.agreement * 100).toFixed(1)}%` : '')))
      if (args.thumbs && r.thumbnail) {
        mkdirSync(args.thumbs, { recursive: true })
        writeFileSync(`${args.thumbs}/${sample}-${name}.jpg`, Buffer.from(r.thumbnail.split(',')[1], 'base64'))
      }
    }
  }
  await browser.close(); await server.close(); clearInterval(busyTimer)
  process.exit(0)
}

// ---- interleaved runs -------------------------------------------------------
const rounds = Number(args.runs)
const reports = []
const discarded = []
for (const sample of args.samples.split(',')) {
  for (const mode of args.modes.split(',')) {
    const results = Object.fromEntries(names.map((n) => [n, { name: n, runs: [] }]))
    const startedAt = new Date().toISOString()
    for (let round = 0; round < rounds; round++) {
      const order = names.map((_, i) => names[(i + round) % names.length])
      await load({ sample })   // fresh page per round
      console.log(`\n${sample} ${mode} round ${round + 1}/${rounds}`)
      for (const name of order) {
        for (let attempt = 0; ; attempt++) {
          await waitIdle()
          const r = await page.evaluate(([n, s, o]) => window.bench.runOne(n, s, o), [name, sample, { mode, output: args.output }])
          r.round = round
          if (r.t0 && overlapsBusy(r.t0, r.t1) && attempt < 5) {
            console.log(`  ${name}: overlapped background load, discarded (attempt ${attempt + 1})`)
            discarded.push({ sample, mode, ...r })
            continue
          }
          if (r.t0 && overlapsBusy(r.t0, r.t1)) r.overlappedBusy = true
          const res = results[name]
          for (const k of ['label', 'maskRes', 'dtype', 'preset', 'error', 'skipped']) if (r[k] !== undefined) res[k] = r[k]
          if (!r.error && !r.skipped) res.runs.push(r)
          const v = mode === 'model' ? `${r.medianMs?.toFixed(2)} ms` : `${r.fps?.toFixed(1)} fps`
          console.log(`  ${name.padEnd(30)} ${r.error ? `ERROR ${r.error}` : r.skipped ?? v}${r.counts ? `  counts ${JSON.stringify(r.counts)}` : ''}`)
          break
        }
      }
    }
    const env = await page.evaluate(() => window.bench.environment())
    for (const res of Object.values(results)) {
      if (!res.runs.length) continue
      const xs = res.runs.map((x) => (mode === 'model' ? x.medianMs : x.fps))
      Object.assign(res, mode === 'model'
        ? { medianMs: median(xs), minMs: Math.min(...xs), maxMs: Math.max(...xs) }
        : { medianFps: median(xs), minFps: Math.min(...xs), maxFps: Math.max(...xs) })
    }
    const report = {
      config: { sample, mode, output: mode === 'transcode' ? args.output : undefined, rounds, order: 'interleaved, rotated per round', startedAt, endedAt: new Date().toISOString() },
      environment: env,
      results: Object.values(results),
    }
    reports.push(report)
    console.log(`\n${sample} ${mode}: median (min–max) over ${rounds} rounds`)
    for (const r of report.results) {
      if (r.error) { console.log(`${r.name.padEnd(30)}ERROR ${r.error}`); continue }
      const [med, lo, hi] = mode === 'model' ? [r.medianMs, r.minMs, r.maxMs] : [r.medianFps, r.minFps, r.maxFps]
      if (med == null) continue
      console.log(r.name.padEnd(30) + [med, lo, hi].map((f) => f.toFixed(mode === 'model' ? 2 : 1).padStart(9)).join('') + `  mask ${r.maskRes ?? '-'}`)
    }
  }
}

clearInterval(busyTimer)
const out = args.out ?? `results/${new Date().toISOString().slice(0, 10)}-${process.platform}-${os.arch()}.json`
mkdirSync('results', { recursive: true })
host.loadavgAtEnd = os.loadavg()
if (process.platform === 'darwin') host.thermalAtEnd = sh('pmset -g therm')
writeFileSync(out, JSON.stringify({
  host, versions, sdkSource, args, reports, discarded,
  busyCmd: args['busy-cmd'] ?? null, busySamples: busy.filter((s) => s.busy).map((s) => s.t),
}, null, 2) + '\n')
console.log(`\nsaved ${out}`)
await browser.close()
await server.close()
