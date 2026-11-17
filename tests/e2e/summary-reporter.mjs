// Playwright reporter that writes a compact results matrix:
//   results/summary.md   scenario × backend requested → per browser: pass/skip/FAIL + backend actually used
//   results/summary.json the same rows, machine-readable
// The tests record what they saw as annotations (requested, backend, denoise…).
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

export default class SummaryReporter {
  constructor(options = {}) {
    this.outputDir = options.outputDir ?? 'results'
    this.rows = new Map()   // test title → { scenario, requested, cells: { project → cell } }
    this.projects = []
    this.notes = []
  }

  onBegin(config, suite) {
    this.projects = config.projects.map((p) => p.name)
    // Rows in source order, whatever order the tests finish in.
    for (const t of suite.allTests()) {
      if (!this.rows.has(t.title)) this.rows.set(t.title, { scenario: t.title.split(' [')[0], requested: t.title.match(/\[(.+)\]$/)?.[1] ?? '', cells: {} })
    }
    this.root = config.configFile ? resolve(config.configFile, '..') : process.cwd()
  }

  onTestEnd(test, result) {
    const project = test.parent.project()?.name ?? '?'
    const ann = (type) => [...test.annotations, ...result.annotations].filter((a) => a.type === type).map((a) => a.description)
    const scenario = test.title.split(' [')[0]
    const requested = test.title.match(/\[(.+)\]$/)?.[1] ?? ''
    const actual = ann('backend').at(-1) ?? ''
    // Retries: the last result wins; a pass after a failure is 'flaky'.
    const status = result.status === 'passed' && result.retry > 0 ? 'flaky' : result.status
    let reason = ''
    if (status === 'skipped') reason = ann('skip').at(-1) ?? ''
    if (status === 'failed' || status === 'timedOut' || status === 'interrupted') {
      reason = (result.error?.message ?? '').replace(/\u001b\[[0-9;]*m/g, '').split('\n').filter(Boolean).slice(0, 2).join(' ')
    }
    const key = test.title
    if (!this.rows.has(key)) this.rows.set(key, { scenario, requested, cells: {} })
    this.rows.get(key).cells[project] = {
      status, actual, reason,
      denoise: ann('denoise').at(-1) ?? '',
      teardown: ann('teardown').at(-1) ?? '',
      ms: result.duration,
    }
  }

  onEnd(result) {
    const dir = resolve(this.root, this.outputDir)
    mkdirSync(dir, { recursive: true })
    const notes = []
    const noteFor = (text) => {
      let i = notes.indexOf(text)
      if (i < 0) { notes.push(text); i = notes.length - 1 }
      return i + 1
    }
    const label = (c) => {
      if (!c) return '–'
      const be = c.actual ? ` (${c.actual})` : ''
      switch (c.status) {
        case 'passed': return `pass${be}`
        case 'flaky': return `pass, flaky${be} [${noteFor('passed on retry')}]`
        case 'skipped': return `skip [${noteFor(c.reason || 'skipped')}]`
        default: return `**FAIL**${be} [${noteFor(c.reason || c.status)}]`
      }
    }
    const projects = this.projects.filter((p) => [...this.rows.values()].some((r) => r.cells[p]))
    const lines = [
      `# Longpipe e2e smoke results`,
      ``,
      `${new Date().toISOString()} · overall: **${result.status}** · cell = result (backend actually used)`,
      ``,
      `| Scenario | Backend requested | ${projects.join(' | ')} |`,
      `|---|---|${projects.map(() => '---').join('|')}|`,
    ]
    // Group by scenario (first appearance), then backend.
    const order = [...new Set([...this.rows.values()].map((r) => r.scenario))]
    const rows = [...this.rows.values()].sort((a, b) => order.indexOf(a.scenario) - order.indexOf(b.scenario))
    for (const r of rows) {
      lines.push(`| ${r.scenario} | ${r.requested} | ${projects.map((p) => label(r.cells[p])).join(' | ')} |`)
    }
    if (notes.length) {
      lines.push('', '**Notes**', '')
      notes.forEach((n, i) => lines.push(`${i + 1}. ${n}`))
    }
    const denoise = [...this.rows.values()].flatMap((r) => projects.filter((p) => r.cells[p]?.denoise).map((p) => `${p}: ${r.cells[p].denoise}`))
    if (denoise.length) {
      lines.push('', '**Denoise**', '')
      for (const d of [...new Set(denoise)]) lines.push(`- ${d}`)
    }
    const md = lines.join('\n') + '\n'
    writeFileSync(resolve(dir, 'summary.md'), md)
    writeFileSync(resolve(dir, 'summary.json'), JSON.stringify([...this.rows.values()], null, 2))
    process.stdout.write(`\n${md}\nsummary written to ${resolve(dir, 'summary.md')}\n`)
  }

  printsToStdio() { return false }
}
