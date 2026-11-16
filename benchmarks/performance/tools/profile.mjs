// Per-dispatch GPU profile of Longpipe networks (and LiteRT.js for reference):
//
//   node tools/profile.mjs <configs.json> <out.json> [--busy-cmd '...'] [--rounds N]
//
// configs.json is an array of web/src/profile.js configs, e.g.
//   [{"impl":"longpipe","backend":"webgpu","net":"mp-square","dtype":"f16"},
//    {"impl":"litert","net":"mp-square"}]
// Each config runs on a freshly loaded page, in headless Chrome with
// --enable-webgpu-developer-features (unquantized timestamp queries); configs are
// interleaved per round (order rotated). --busy-cmd as in bench.mjs: measurements
// that overlap a busy sample are redone.
import { exec } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'vite'
import { chromium } from 'playwright-core'

const [cfgFile, outFile, ...rest] = process.argv.slice(2)
const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d }
const busyCmd = opt('--busy-cmd')
const rounds = Number(opt('--rounds', '1'))
const configs = JSON.parse(readFileSync(cfgFile, 'utf8'))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const busy = []
let timer
if (busyCmd) { const poll = () => exec(busyCmd, (e) => busy.push({ t: Date.now(), busy: !e })); poll(); timer = setInterval(poll, 1000) }
const overlaps = (t0, t1) => busy.some((s) => s.busy && s.t >= t0 - 1500 && s.t <= t1 + 1500)
async function waitIdle() {
  if (!busyCmd) return
  for (let said = false; ;) {
    const recent = busy.filter((s) => s.t > Date.now() - 5000)
    if (recent.length >= 4 && recent.every((s) => !s.busy)) return
    if (!said && recent.some((s) => s.busy)) { console.log('  (busy: waiting)'); said = true }
    await sleep(1000)
  }
}

const root = new URL('../', import.meta.url).pathname
const server = await createServer({ configFile: `${root}vite.config.js`, server: { port: 0 }, logLevel: 'error' })
await server.listen()
const origin = server.resolvedUrls.local[0].replace(/\/$/, '')
const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'],
})
const page = await browser.newPage()
page.on('console', (m) => ['error', 'warning'].includes(m.type()) && !/404|^(INFO|WARNING): |^[IW]\d{4} /.test(m.text()) && console.error('[page]', m.text().slice(0, 300)))
page.on('pageerror', (e) => console.error('[pageerror]', e.message))
page.setDefaultTimeout(0)

const results = []
for (let round = 0; round < rounds; round++) {
  const order = configs.map((_, i) => configs[(i + round) % configs.length])
  for (const cfg of order) {
    for (let attempt = 0; ; attempt++) {
      await waitIdle()
      await page.goto(`${origin}/profile.html`)
      await page.waitForFunction(() => window.ready)
      let r
      try {
        r = await page.evaluate(([fn, c]) => window[fn](c), [cfg.fn ?? 'profile', cfg])
      } catch (e) { r = { cfg, error: String(e.message).slice(0, 500) } }
      if (r.t0 && overlaps(r.t0, r.t1) && attempt < 5) { console.log('  overlapped worker job, redo', JSON.stringify(cfg)); continue }
      r.round = round
      results.push(r)
      const w = r.wall
      console.log(JSON.stringify(cfg), r.error ?? (w ? `lat ${w.latencyMs.toFixed(3)} thr ${w.throughputMs.toFixed(3)} cpu ${w.cpuRunMs.toFixed(3)} ms | ${JSON.stringify(r.counts)}${r.gpu ? ` | gpu 1pass ${r.gpu.onePassMs.toFixed(3)} sum ${r.gpu.sumOfPassesMs.toFixed(3)}` : ''}` : JSON.stringify(r)))
      break
    }
  }
}
writeFileSync(outFile, JSON.stringify({ chrome: browser.version(), results, busySamples: busy.filter((s) => s.busy).map((s) => s.t) }, null, 1))
clearInterval(timer)
await browser.close()
await server.close()
