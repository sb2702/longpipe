#!/usr/bin/env node
// Device self-test results → a markdown table, one row per run.
//
//   node examples/vanilla-webrtc/selftest-table.mjs [results.jsonl] [--json]
//
// Default input: selftest-results.jsonl next to this file (written by the
// dev server's POST /__selftest; see vite.config.js). --json adds each run's
// full JSON below the table, for details the row leaves out.
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { headerRows, toRow } from './selftest-format.js'

const args = process.argv.slice(2)
const file = args.find((a) => !a.startsWith('--')) ?? fileURLToPath(new URL('./selftest-results.jsonl', import.meta.url))
let text
try {
  text = readFileSync(file, 'utf8')
} catch (e) {
  console.error(`no results at ${file} (${e.code}): run the self-test first (README, "Device self-test")`)
  process.exit(1)
}
const runs = text.split('\n').filter(Boolean).flatMap((line, i) => {
  try { return [JSON.parse(line)] } catch { console.error(`line ${i + 1}: not JSON, skipped`); return [] }
})
console.log(headerRows())
for (const r of runs) console.log(toRow(r))
const sdks = [...new Set(runs.map((r) => r.env?.sdk).filter(Boolean))]
console.log(`\n${runs.length} run(s), SDK ${sdks.join(', ') || '?'}; ${runs[0]?.receivedAt?.slice(0, 10) ?? ''} – ${runs.at(-1)?.receivedAt?.slice(0, 10) ?? ''}`)
if (args.includes('--json')) for (const r of runs) console.log('\n```json\n' + JSON.stringify(r, null, 2) + '\n```')
