// Engine profiler: per-dispatch GPU time and CPU/GPU split for one network.
// Driven by tools/profile.mjs (see there); window.profile(cfg) builds one network
// and returns
//   wall:   latency (run + wait, median), throughput (back-to-back runs) and CPU
//           time inside run(),
//   counts: queue submits / passes / dispatches per run,
//   gpu:    per-dispatch GPU time (timestamp queries, each captured dispatch
//           replayed in its own pass) and the whole run replayed in ONE pass.
// Dispatches are captured by wrapping the WebGPU API, so it works for any engine
// on the page (Longpipe, LiteRT.js).
//
// cfg: { impl: 'longpipe'|'litert', backend: 'webgpu'|'webgl',
//        net: 'mp-square'|'mp-landscape'|'xs'|'small'|'medium'|'large'|'xl',
//        dtype: 'f16'|'f32', iters?, perOp?, frame?: {w, h}, exclude?: {name: labelRegex} }
// frame: profile the whole per-frame path (VideoFrame → Input → network →
// RenderOp upscale + composite) at that frame size instead of the network alone.
// exclude: also report the one-pass time with matching dispatches left out (an
// upper bound on what fusing them away could save).
import * as LP from '~/model.ts'
import { WebGPUOp } from '~/model/backends/webgpu/base_webgpu_op.ts'

const median = (xs) => { const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2 }
const pct = (xs, p) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(xs.length * p))]

// ---- dispatch capture (patches the WebGPU API; works for any engine) --------
const CP = GPUComputePassEncoder.prototype, CE = GPUCommandEncoder.prototype, Q = GPUQueue.prototype
const orig = {
  setPipeline: CP.setPipeline, setBindGroup: CP.setBindGroup, dispatch: CP.dispatchWorkgroups,
  dispatchIndirect: CP.dispatchWorkgroupsIndirect, beginCompute: CE.beginComputePass, beginRender: CE.beginRenderPass,
  submit: Q.submit, writeBuffer: Q.writeBuffer,
}
let rec = null
let label = ''
const counts = { submits: 0, computePasses: 0, renderPasses: 0, dispatches: 0, writeBuffers: 0 }
const st = new WeakMap()
CP.setPipeline = function (p) { if (rec) (st.get(this) ?? st.set(this, { bg: {} }).get(this)).p = p; return orig.setPipeline.call(this, p) }
CP.setBindGroup = function (i, g, ...rest) {
  if (rec) (st.get(this) ?? st.set(this, { bg: {} }).get(this)).bg[i] = { g, rest }
  return orig.setBindGroup.call(this, i, g, ...rest)
}
CP.dispatchWorkgroups = function (x, y = 1, z = 1) {
  counts.dispatches++
  if (rec) { const s = st.get(this); rec.push({ label, p: s.p, bg: { ...s.bg }, args: [x, y, z] }) }
  return orig.dispatch.call(this, x, y, z)
}
CP.dispatchWorkgroupsIndirect = function (buf, off) {
  counts.dispatches++
  if (rec) { const s = st.get(this); rec.push({ label: label + ' (indirect)', p: s.p, bg: { ...s.bg }, indirect: [buf, off] }) }
  return orig.dispatchIndirect.call(this, buf, off)
}
CE.beginComputePass = function (d) { counts.computePasses++; return orig.beginCompute.call(this, d) }
CE.beginRenderPass = function (d) { counts.renderPasses++; return orig.beginRender.call(this, d) }
Q.submit = function (b) { counts.submits++; return orig.submit.call(this, b) }
Q.writeBuffer = function (...a) { counts.writeBuffers++; return orig.writeBuffer.apply(this, a) }
const resetCounts = () => { for (const k in counts) counts[k] = 0 }

function labelOps(Op) {
  const run = Op.prototype.run
  Op.prototype.run = function () {
    const prev = label
    const o = this.output
    label = `${this.constructor.name.replace(/WebGPU$/, '')} ${o ? `${o.w}x${o.h}x${o.c}` : ''}`
    try { return run.call(this) } finally { label = prev }
  }
}
labelOps(WebGPUOp)

function replay(pass, d) {
  pass.setPipeline(d.p)
  for (const [i, { g, rest }] of Object.entries(d.bg)) pass.setBindGroup(Number(i), g, ...rest)
  if (d.indirect) pass.dispatchWorkgroupsIndirect(...d.indirect)
  else pass.dispatchWorkgroups(...d.args)
}

// GPU time per dispatch (each in its own timestamped pass) + the whole captured
// graph replayed in ONE pass (pure GPU time, no per-pass overhead).
async function timeDispatches(device, list, frames = 30, warmup = 5) {
  const n = list.length
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 * n + 2 })
  const resolve = device.createBuffer({ size: (2 * n + 2) * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
  const read = device.createBuffer({ size: (2 * n + 2) * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const per = new Array(n).fill(0)
  const singles = []
  for (let f = 0; f < warmup + frames; f++) {
    const enc = device.createCommandEncoder()
    let pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * n, endOfPassWriteIndex: 2 * n + 1 } })
    for (const d of list) replay(pass, d)
    pass.end()
    list.forEach((d, i) => {
      pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } })
      replay(pass, d)
      pass.end()
    })
    enc.resolveQuerySet(qs, 0, 2 * n + 2, resolve, 0)
    enc.copyBufferToBuffer(resolve, 0, read, 0, (2 * n + 2) * 8)
    device.queue.submit([enc.finish()])
    await read.mapAsync(GPUMapMode.READ)
    const t = new BigInt64Array(read.getMappedRange().slice(0))
    read.unmap()
    if (f < warmup) continue
    for (let i = 0; i < n; i++) per[i] += Number(t[2 * i + 1] - t[2 * i]) / 1e6 / frames
    singles.push(Number(t[2 * n + 1] - t[2 * n]) / 1e6)
  }
  qs.destroy(); resolve.destroy(); read.destroy()
  const byLabel = {}
  list.forEach((d, i) => {
    const k = d.label.split(' ')[0] || '?'
    const e = (byLabel[k] ??= { dispatches: 0, ms: 0 })
    e.dispatches++; e.ms += per[i]
  })
  return {
    sumOfPassesMs: per.reduce((a, b) => a + b, 0),
    onePassMs: median(singles),
    byKind: Object.fromEntries(Object.entries(byLabel).sort((a, b) => b[1].ms - a[1].ms)),
    perDispatch: list.map((d, i) => ({ label: d.label, wg: d.args ?? 'indirect', ms: +per[i].toFixed(4) })),
  }
}

async function makeDevice() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  const requiredFeatures = ['timestamp-query', 'shader-f16'].filter((f) => adapter.features.has(f))
  const device = await adapter.requestDevice({ requiredFeatures })
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message))
  return { adapter, device }
}

const channel = new MessageChannel()
const yieldTask = () => new Promise((r) => { channel.port1.onmessage = r; channel.port2.postMessage(0) })
async function glWait(gl) {
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
  gl.flush()
  for (;;) {
    const s = gl.clientWaitSync(fence, 0, 0)
    if (s === gl.ALREADY_SIGNALED || s === gl.CONDITION_SATISFIED) break
    if (s === gl.WAIT_FAILED) throw new Error('clientWaitSync failed')
    await yieldTask()
  }
  gl.deleteSync(fence)
}

const TIER_FILE = (t, dtype) => `/assets/models/model_${t}${dtype === 'f16' ? '.f16' : ''}.bin`
const MP = { 'mp-square': { file: 'mediapipe_selfie_segmenter.f16.bin', w: 256, h: 256, tflite: 'selfie_segmenter.tflite' },
             'mp-landscape': { file: 'mediapipe_selfie_segmenter_landscape.f16.bin', w: 256, h: 144, tflite: 'selfie_segmenter_landscape.tflite' } }

function inputData(h, w) {
  const d = new Float32Array(h * w * 4)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * w + x) * 4
    d[p] = 0.5 + 0.5 * Math.sin(x * 0.07 + y * 0.03); d[p + 1] = (x / w); d[p + 2] = (y / h) * 0.8 + 0.1; d[p + 3] = 1
  }
  return d
}

async function buildLongpipe(cfg) {
  const lib = LP
  const canvas = new OffscreenCanvas(1, 1)
  let device = null
  let backend
  if (cfg.backend === 'webgl') backend = await lib.WebGLBackend.create({ canvas, dtype: cfg.dtype })
  else { ({ device } = await makeDevice()); backend = await lib.WebGPUBackend.create({ canvas, dtype: cfg.dtype, device }) }
  let model, h, w
  // Frame mode (cfg.frame = {w, h}): the full per-frame path the bench runs —
  // VideoFrame → Input op → network → RenderOp (upscale + display input + composite).
  if (cfg.frame) { canvas.width = cfg.frame.w; canvas.height = cfg.frame.h }
  ;({ h, w } = MP[cfg.net] ?? lib.TIER_CONFIG[cfg.net].canvasRes)
  const netIn = cfg.frame ? backend.ops.Input(h, w) : null
  const x = netIn ? netIn.output : backend.tensor(h, w, 4, inputData(h, w))
  if (MP[cfg.net]) {
    const weights = lib.loadWeightsFromBinary(await (await fetch(`/assets/models/${MP[cfg.net].file}`)).arrayBuffer())
    model = new lib.MediaPipeSelfie(backend, x, weights)
  } else {
    const tc = lib.TIER_CONFIG[cfg.net]
    const weights = lib.loadWeightsFromBinary(await (await fetch(TIER_FILE(cfg.net, cfg.dtype))).arrayBuffer())
    model = new lib.TierModel(backend, x, weights.base, weights.wrapper, tc.wrapper, tc.base)
  }
  if (cfg.frame) {
    const src = new OffscreenCanvas(cfg.frame.w, cfg.frame.h)
    const g = src.getContext('2d'); g.fillStyle = '#468'; g.fillRect(0, 0, cfg.frame.w, cfg.frame.h)
    g.fillStyle = '#eca'; g.beginPath(); g.arc(cfg.frame.w / 2, cfg.frame.h / 2, cfg.frame.h / 3, 0, 7); g.fill()
    const vf = new VideoFrame(src, { timestamp: 0 })
    const render = new lib.RenderOp(backend)
    render.attachNetwork(model, netIn, { upscaler: 'bilinear', background: { mode: 'solid', color: [0, 0.7, 0.25] } })
    return {
      device, backend, model,
      run: () => { render.setSource(vf); render.run() },
      sync: () => (cfg.backend === 'webgl' ? glWait(backend.gl) : backend.sync()),
      destroy: () => { vf.close(); backend.destroy() },
    }
  }
  return {
    device, backend, model,
    run: () => model.run(),
    sync: () => (cfg.backend === 'webgl' ? glWait(backend.gl) : backend.sync()),
    read: () => backend.readback(model.output),
    destroy: () => backend.destroy(),
  }
}

async function buildLiteRt(cfg) {
  const { device } = await makeDevice()
  const litert = await import('@litertjs/core')
  await litert.loadLiteRt('/vendor/litert/', { threads: self.crossOriginIsolated })
  litert.setWebGpuDevice(device)
  const { w: W, h: H, tflite } = MP[cfg.net]
  const model = await litert.loadAndCompile(`/assets/models/${tflite}`, { accelerator: 'webgpu' })
  const buf = device.createBuffer({ size: W * H * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
  const input = new litert.Tensor(buf, [1, H, W, 3], 'float32')
  const m = {
    device,
    // Keep the latest output alive (a captured run is replayed after it).
    run: async () => { const [out] = await model.run([input]); m.last?.delete(); m.last = out },
    sync: () => device.queue.onSubmittedWorkDone(),
    destroy: () => { m.last?.delete(); model.delete?.() },
  }
  return m
}

async function wall(m, iters) {
  for (let i = 0; i < 30; i++) await m.run()
  await m.sync()
  const lat = []
  for (let i = 0; i < iters; i++) {
    const t = performance.now()
    await m.run()
    await m.sync()
    lat.push(performance.now() - t)
  }
  // Throughput: back-to-back runs, one sync at the end; also CPU time inside run().
  let cpu = 0
  const t0 = performance.now()
  for (let i = 0; i < iters; i++) { const t = performance.now(); await m.run(); cpu += performance.now() - t }
  await m.sync()
  const thr = (performance.now() - t0) / iters
  return { latencyMs: median(lat), latencyP10: pct(lat, 0.1), latencyP90: pct(lat, 0.9), throughputMs: thr, cpuRunMs: cpu / iters }
}

window.profile = async (cfg) => {
  const iters = cfg.iters ?? 200
  const m = cfg.impl === 'litert' ? await buildLiteRt(cfg) : await buildLongpipe(cfg)
  const r = { cfg }
  try {
    await m.run(); await m.sync()
    r.t0 = Date.now()
    r.wall = await wall(m, iters)
    // Counts + capture over one run (the last one: its buffers stay alive).
    resetCounts()
    rec = []
    await m.run()
    const captured = rec
    rec = null
    await m.sync()
    r.counts = { ...counts }
    if (m.device && cfg.perOp !== false && captured.length) {
      r.gpu = await timeDispatches(m.device, captured)
      // What-if: one-pass time with some dispatches left out (upper bound on
      // what fusing them away could save).
      if (cfg.exclude) {
        r.excluded = {}
        for (const [name, re] of Object.entries(cfg.exclude)) {
          const rx = new RegExp(re)
          const keep = captured.filter((d) => !rx.test(d.label))
          r.excluded[name] = { dropped: captured.length - keep.length, onePassMs: (await timeDispatches(m.device, keep)).onePassMs }
        }
      }
    }
    r.t1 = Date.now()
  } finally {
    m.destroy()
  }
  return r
}

document.getElementById('out').textContent = 'ready'
window.ready = true
