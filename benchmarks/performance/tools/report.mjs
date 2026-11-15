// Turns a results JSON written by tools/bench.mjs into markdown tables.
//
//   node tools/report.mjs results/2026-11-15-apple-m4.json > results/2026-11-15-apple-m4.md
import { readFileSync } from 'node:fs'

const file = process.argv[2]
if (!file) throw new Error('usage: node tools/report.mjs <results.json>')
const { host, versions, reports } = JSON.parse(readFileSync(file, 'utf8'))

const find = (mode, height) => reports.find((r) => r.config.mode === mode && r.config.height === height)
const heights = [...new Set(reports.map((r) => r.config.height))].sort((a, b) => a - b)
const names = [...new Set(reports.flatMap((r) => r.results.map((x) => x.name)))]
const label = (name) => reports.flatMap((r) => r.results).find((x) => x.name === name).label
const res = (mode, h, name) => find(mode, h)?.results.find((x) => x.name === name)
const fmt = (x, d) => (x == null ? '–' : x.toFixed(d))
const spread = (r, d) => (r && !r.error && r.runs.length ? ` (${fmt(r.minFps ?? r.minMs, d)}–${fmt(r.maxFps ?? r.maxMs, d)})` : '')

const out = []
out.push('| Pipeline | Mask (model output) | ' + heights.map((h) => `${h}p transcode FPS`).join(' | ') + ' | ' +
  heights.map((h) => `${h}p process FPS`).join(' | ') + ' | ' + heights.map((h) => `${h}p model ms`).join(' | ') + ' |')
out.push('|---|---|' + '---:|'.repeat(heights.length * 3))
for (const name of names) {
  const any = reports.flatMap((r) => r.results).find((x) => x.name === name && !x.error)
  const cells = [
    ...heights.map((h) => { const r = res('transcode', h, name); return r?.error ? 'error' : fmt(r?.medianFps, 0) + spread(r, 0) }),
    ...heights.map((h) => { const r = res('process', h, name); return r?.error ? 'error' : fmt(r?.medianFps, 0) }),
    ...heights.map((h) => { const r = res('model', h, name); return r?.error ? 'error' : fmt(r?.medianMs, 2) }),
  ]
  out.push(`| ${label(name)} | ${any?.maskRes ?? '–'}${any?.deliveredMaskRes ? ', returned upsampled to frame size' : ''} | ${cells.join(' | ')} |`)
}
const env = reports[0].environment
console.log(`Host: ${host.chip ?? host.cpu} (${host.model ?? ''}), ${host.memoryGB} GB, macOS ${host.macos ?? host.release}; Chrome ${host.chrome}${host.headless ? ' (headless)' : ''}`)
console.log(`WebGPU adapter: ${[env.webgpu?.vendor, env.webgpu?.architecture].filter(Boolean).join(' ')}; WebGL: ${env.webglRenderer}; cross-origin isolated: ${env.crossOriginIsolated}`)
console.log(`Power: ${(host.power ?? '').replace(/\s+/g, ' ')}; ${host.lowPowerMode?.replace(/\s+/g, ' ') ?? ''}`)
console.log(`Versions: ${Object.entries(versions).map(([k, v]) => `${k} ${v}`).join(', ')}`)
console.log(`Runs: ${reports[0].config.runs} timed (median shown, min–max in brackets for transcode); model mode ${reports.find((r) => r.config.mode === 'model')?.config.modelIterations ?? '–'} iterations per run\n`)
console.log(out.join('\n'))
