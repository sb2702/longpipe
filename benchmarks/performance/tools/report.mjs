// Turns results JSON written by tools/bench.mjs into markdown tables, one
// column per (sample, session).
//
//   node tools/report.mjs s1=results/a.json s2=results/b.json [diag=results/c.json] > tables.md
//
// Transcode reports become the main table; process/model reports a diagnostic
// table; shipped-* runs a key-frame / flow-frame count table.
import { readFileSync } from 'node:fs'

const inputs = process.argv.slice(2).map((a) => {
  const [label, file] = a.includes('=') ? a.split('=') : [a, a]
  return { label, ...JSON.parse(readFileSync(file, 'utf8')) }
})
if (!inputs.length) throw new Error('usage: node tools/report.mjs label=results.json ...')

const fmt = (x, d = 0) => (x == null ? '–' : x.toFixed(d))
const GROUPS = [
  ['Ceilings (no model)', (n) => n === 'passthrough' || n.startsWith('canvas-copy')],
  ['1. MediaPipe model, MediaPipe Tasks engine, every frame', (n) => n.startsWith('mediapipe-')],
  ['2. MediaPipe model, LiteRT.js engine, every frame', (n) => n.startsWith('litert-')],
  ['3. MediaPipe model, Longpipe engine, every frame', (n) => n.startsWith('lp-mediapipe-')],
  ['4. Longpipe model, Longpipe engine, network every frame', (n) => n.startsWith('longpipe-')],
  ['5. Longpipe model, Longpipe engine, as shipped (key + flow frames)', (n) => n.startsWith('shipped-')],
]

// All (report, input) pairs for a mode.
const reportsFor = (mode) => inputs.flatMap((inp) => inp.reports.filter((r) => r.config.mode === mode).map((r) => ({ inp, r })))
const names = [...new Set(inputs.flatMap((i) => i.reports.flatMap((r) => r.results.map((x) => x.name))))]
const label = (n) => inputs.flatMap((i) => i.reports.flatMap((r) => r.results)).find((x) => x.name === n && x.label)?.label ?? n
const mask = (n) => inputs.flatMap((i) => i.reports.flatMap((r) => r.results)).find((x) => x.name === n && x.maskRes)?.maskRes ?? '–'

function table(mode, cell) {
  const cols = reportsFor(mode).sort((a, b) => a.r.config.sample.localeCompare(b.r.config.sample, 'en', { numeric: true }))
  if (!cols.length) return ''
  const out = [`| Pipeline | Mask | ${cols.map(({ inp, r }) => `${r.config.sample} ${inp.label}`).join(' | ')} |`, `|---|---|${'---:|'.repeat(cols.length)}`]
  for (const [title, test] of GROUPS) {
    const rows = names.filter(test)
    if (!rows.length) continue
    out.push(`| **${title}** | | ${cols.map(() => '').join(' | ')} |`)
    for (const n of rows.filter((n) => cols.some(({ r }) => r.results.find((x) => x.name === n && (x.runs.length || x.error))))) out.push(`| ${label(n)} | ${mask(n).replace('x', '×')} | ${cols.map(({ r }) => cell(r.results.find((x) => x.name === n))).join(' | ')} |`)
  }
  return out.join('\n')
}

const fps = (x) => (!x ? '–' : x.error ? 'error' : x.medianFps == null ? '–' : `${fmt(x.medianFps)} (${fmt(x.minFps)}–${fmt(x.maxFps)})`)
const ms = (x) => (!x ? '–' : x.error ? 'error' : x.medianMs == null ? '–' : `${fmt(x.medianMs, 2)} (${fmt(x.minMs, 2)}–${fmt(x.maxMs, 2)})`)

for (const inp of inputs) {
  const env = inp.reports[0].environment
  const h = inp.host
  console.log(`- **${inp.label}**: ${inp.reports.map((r) => `${r.config.sample} ${r.config.mode}${r.config.output ? ` (${r.config.output})` : ''} ${r.config.startedAt.slice(11, 16)}–${r.config.endedAt.slice(11, 16)} UTC`).join('; ')}. ` +
    `${inp.reports[0].config.rounds} rounds. Discarded (overlapped a worker job) and repeated: ${inp.discarded?.length ?? 0}. ` +
    `Chrome ${h.chrome}${h.headless ? ' headless' : ''}; ${env.webgpu?.vendor} ${env.webgpu?.architecture}; load avg start ${h.loadavgAtStart?.map((x) => x.toFixed(1)).join('/')}, end ${h.loadavgAtEnd?.map((x) => x.toFixed(1)).join('/')}.`)
}
console.log('\n### Transcode FPS: median (min–max)\n')
console.log(table('transcode', fps))
const proc = table('process', fps)
if (proc) console.log('\n### Diagnostic: process FPS (decode → effect → wait for output pixels, no encoder)\n\n' + proc)
const model = table('model', ms)
if (model) console.log('\n### Diagnostic: model-only ms per frame, GPU-synchronised\n\n' + model)

// Key / flow frame counts for the as-shipped rows.
const counted = inputs.flatMap((i) => i.reports.flatMap((r) => r.results.flatMap((x) => x.runs.filter((y) => y.counts).map((y) => ({ name: x.name, preset: x.preset, ...y.counts })))))
if (counted.length) {
  console.log('\n### As shipped: model runs per 192-frame timed pass (every timed run)\n')
  console.log('| Pipeline | skipFrames | flow wired | frames | full-network runs | flow-network runs | runs checked |')
  console.log('|---|---:|---|---:|---:|---:|---:|')
  for (const n of [...new Set(counted.map((c) => c.name))]) {
    const cs = counted.filter((c) => c.name === n)
    const uniq = [...new Set(cs.map((c) => `${c.frames}|${c.network}|${c.flow}`))]
    const [f, net, fl] = uniq.length === 1 ? uniq[0].split('|') : ['varies', 'varies', 'varies']
    console.log(`| ${n} | ${cs[0].preset?.skipFrames} | ${cs[0].preset?.flowWired ? 'yes' : 'no'} | ${f} | ${net} | ${fl} | ${cs.length} |`)
  }
}
