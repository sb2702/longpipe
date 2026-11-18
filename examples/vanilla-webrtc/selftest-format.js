// One self-test result (selftest.js) → one markdown table row. Shared by the
// page and by selftest-table.mjs (Node), so no DOM here.

export const HEADER = [
  'Browser', 'OS', 'GPU', 'Backend', 'WebRTC video', 'Audio',
  'GPU loss → recovered', 'Worker crash → recovered', 'No-GPU fallback',
]

const mark = (ok) => (ok ? '✓' : '✗')
const sec = (ms) => (ms == null ? '?' : `${(ms / 1000).toFixed(1)} s`)
const cell = (s) => String(s ?? '–').replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim()
const failed = (st) => `✗ ${st?.error ? cell(st.error).slice(0, 60) : 'not run'}`

// GPU in a few words: "Apple M2 (metal-3)", "NVIDIA GeForce RTX 4090" …
export function gpuLabel(gpu) {
  if (!gpu) return '?'
  const a = gpu.webgpu
  if (a && (a.description || a.vendor || a.architecture)) {
    return a.description || [a.vendor, a.architecture].filter(Boolean).join(' ')
  }
  const r = gpu.webglRenderer
  if (!r) return '?'
  // "ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 (0x2684) Direct3D11 vs_5_0 ps_5_0, D3D11)" → the name
  const m = /^ANGLE \([^,]+, (.+?)(?: \(0x[0-9a-f]+\))?(?:,| Direct3D| OpenGL| Vulkan| Metal)/i.exec(r)
  return m ? m[1] : r
}

function recovery(st, { worker = false } = {}) {
  if (!st || st.error) return failed(st)
  if (!st.recovered) return `✗ no onRecovered in 20 s${st.framesDuring > 0 ? ', video kept flowing' : ', video stalled'}`
  const parts = [sec(st.recoverMs), `${st.backendBefore} → ${st.backendAfter}`]
  if (worker) {
    parts.push(st.trackChanged
      ? `replaced (replaceTrack ${st.replaceTrack?.ok ? 'ok' : 'failed'})`
      : 'same track')
  }
  if (!(st.framesDuring > 0 && st.framesAfter > 0)) parts.push(`frames ${st.framesDuring > 0 ? '✓' : '✗'} during / ${st.framesAfter > 0 ? '✓' : '✗'} after`)
  else if (st.maxStallMs > 500) parts.push(`max stall ${sec(st.maxStallMs)}`)
  if (st.effectAfter === false) parts.push('effect ✗ after')
  if (st.rebuiltFirst) parts.push('fresh pipeline')
  return `${mark(st.ok)} ${parts.join(', ')}`
}

export function toCells(r) {
  const e = r.env ?? {}
  const s = r.steps ?? {}
  const browser = `${e.browser ?? '?'} ${e.browserVersion ?? ''}`.trim() + (e.ios ? ' (WebKit)' : '')
  const st = s.startup
  // e.g. "webgl2 (xs, f16), WebGPU lost ×2 at start"
  const lost = (st?.errors ?? []).filter((x) => x.type === 'backend-lost').length
  const backend = st?.ready
    ? `${st.ready.backend}${st.ready.preset ? ` (${st.ready.preset}${st.ready.dtype ? `, ${st.ready.dtype}` : ''})` : ''}` +
      (lost ? `, GPU lost ×${lost} at start` : '') + (st.readyMs != null ? `, ready ${sec(st.readyMs)}` : '')
    : failed(st)
  const v = s.video
  const video = v && !v.error
    ? `${mark(v.ok)} ${v.remoteFps} fps ${v.width}×${v.height}, effect ${mark(v.effect)}`
    : failed(v)
  const a = s.audio
  const audio = a && !a.error
    ? `${mark(a.ok)} ${a.denoiseModel ?? 'no denoise'}${a.denoiseModel ? (a.denoiseActive ? ' active' : ' inactive') : ''}`
    : failed(a)
  const n = s.noGpu
  const noGpu = n && !n.error
    ? `${mark(n.ok)} ${n.framesDelta > 0 ? 'raw video' : 'no video'}, no-backend ${n.noBackendError ? (n.noBackendFatal ? 'fatal' : 'non-fatal') : 'missing'}`
    : failed(n)
  return [browser, e.os ?? '?', gpuLabel(e.gpu), backend, video, audio,
    recovery(s.gpuLoss), recovery(s.workerCrash, { worker: true }), noGpu].map(cell)
}

export const toRow = (r) => `| ${toCells(r).join(' | ')} |`
export const headerRows = () => `| ${HEADER.join(' | ')} |\n|${HEADER.map(() => '---').join('|')}|`
