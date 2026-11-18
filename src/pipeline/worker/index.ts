// Worker entry point. Responsibilities:
//
//   1. Dispatch incoming control-plane messages to typed handlers. A handler
//      that throws answers THAT request with an error (main rejects the
//      call's promise); it no longer takes the pipeline down.
//   2. handleInit: setupBackend → construct Renderer in boot mode →
//      wire input + output + start pipe (passthrough flowing immediately)
//      → resolvePreset (autotune; pipe keeps pumping passthrough during
//      this) → return InitResponse.
//      No usable GPU → 'none' mode: no renderer. On transfer-capture the
//      worker draws raw frames with a 2D context; on the other paths main
//      does raw passthrough, so the worker just releases its input.
//   3. handleStartRender: renderer.setPreset(preset, weights) — attaches
//      the network in place. Renderer flips from passthrough to effect
//      mid-pipe; no pipe restart, no stream relock.
//   4. Health: a heartbeat every TUNING.heartbeatMs (main's watchdog), and
//      events for GPU loss ('error' type 'backend-lost'), pipe failure
//      ('error' type 'worker-crash') and input end ('inputEnded'). Main
//      reacts by switching the output to raw and restarting this worker.
//
// All actual work lives in the submodules; this file should stay thin.

import { createLogger, setDebug } from '../debug'

const log = createLogger('worker')
log('script loaded')

import type {
  WorkerRequest,
  WorkerResponse,
  WorkerEvent,
  CmdName,
  CmdDataMap,
  InitData,
  InitResponse,
  PresetSwapResult,
  EventName,
  EventMap,
  RendererStats,
} from '../messages'
import type { Background } from '../background'
import type { ManualPreset, PresetName } from '../presets'
import { resolveNamedPreset } from '../presets'
import { Renderer } from './renderer'
import { setupBackend }     from './setup_backend'
import { resolvePreset }    from './resolve_preset'
import { createInputStream } from './create_input'
import { createOutputSink, createRawOutputSink } from './create_output'
import { recordLatency, meanLatencyMs } from './latency'

let renderer:  Renderer        | null = null
let initData:  InitData        | null = null

// Input pipe. The output sink lives for the worker's lifetime; the input
// side can be swapped (setInput) — hence pump() leaves the sink open, and a
// generation counter so a superseded pipe's settle is ignored.
let outputSink:  WritableStream<VideoFrame> | null = null
let inputAbort:  AbortController | null = null
let pipeDone:    Promise<void> | null = null
let pipeGen = 0

// Teardown / failure flags: GPU loss and our own destroy both make the pipe
// fail; neither should be reported as a pipe crash on top.
let shuttingDown = false
let backendLost  = false

// No-GPU mode (resolvedBackend 'none').
let noneMode = false

// Frames written to the output (effect or raw) — reported in the heartbeat.
let framesOut = 0
const frameTimes: number[] = []
const onFrame = (entry?: number) => {
  framesOut++
  recordLatency(entry)
  if (noneMode) {
    const now = performance.now()
    frameTimes.push(now)
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift()
  }
}
let heartbeat: ReturnType<typeof setInterval> | null = null

// State held between handleInit and handleStartRender — the latter only
// needs to know which preset the autotune chose (so it can pass it to
// renderer.setPreset alongside the weights main fetched).
let resolvedPreset: ManualPreset | null = null
// Model resolved on main (InitData.model). MediaPipe: no preset / autotune.
let model: import('../model_choice').ResolvedModel = { family: 'longpipe' }


self.onmessage = async function (event: MessageEvent<WorkerRequest>) {
  const { cmd, data, request_id } = event.data
  log('cmd received:', cmd, 'request_id:', request_id)
  try {
    const res = await handleCommand(cmd, data)
    log('cmd ok:', cmd)
    respond(request_id, res)
  } catch (err) {
    log('cmd FAILED:', cmd, err)
    respondError(request_id, `${cmd} failed: ${(err as Error)?.message ?? String(err)}`)
  }
}

// Stray async failures (e.g. a background-video port callback) shouldn't
// vanish silently.
self.addEventListener('unhandledrejection', (e) => {
  console.error('[longpipe/worker] unhandled rejection:', (e as PromiseRejectionEvent).reason)
})

async function handleCommand(cmd: CmdName, data: unknown): Promise<unknown> {
  switch (cmd) {
    case 'init':          return handleInit(data as InitData)
    case 'startRender':   return handleStartRender(data as CmdDataMap['startRender'])
    case 'setBackground': return handleSetBackground(data as Background)
    case 'setEnabled':    return handleSetEnabled((data as CmdDataMap['setEnabled']).enabled)
    case 'setPreset':   return handleSetPreset(data as CmdDataMap['setPreset'])
    case 'getStats':    return getStats()
    case 'destroy':     return handleDestroy()
    case 'attachPreview': return handleAttachPreview(data as CmdDataMap['attachPreview'])
    case 'setPreview':    return handleSetPreview(data as CmdDataMap['setPreview'])
    case 'clearPreview':  return handleClearPreview()
    case 'setTouchup':    return handleSetTouchup(data as CmdDataMap['setTouchup'])
    case 'setReframe':    return handleSetReframe(data as CmdDataMap['setReframe'])
    case 'reframeNow':    return handleReframeNow()
    case 'setInput':      return handleSetInput(data as CmdDataMap['setInput'])
    case 'debug':         return handleDebug(data as CmdDataMap['debug'])
    default: throw new Error(`unknown cmd: ${cmd}`)
  }
}

function startHeartbeat(ms: number): void {
  if (heartbeat) clearInterval(heartbeat)
  heartbeat = setInterval(() => emit('heartbeat', { framesOut }), ms)
}

// (Re)start the input → output pipe. The sink stays open across input swaps.
function startInputPipe(input: ReadableStream<VideoFrame>, sink: WritableStream<VideoFrame>): void {
  const gen = ++pipeGen
  const abort = new AbortController()
  inputAbort = abort
  pipeDone = pump(input, sink, abort.signal)
    .then(() => {
      if (gen !== pipeGen || shuttingDown) return
      log('pipe completed (input ended)')
      emit('inputEnded', undefined)
    })
    .catch(err => {
      if (gen !== pipeGen || abort.signal.aborted || shuttingDown || backendLost) {
        log('pipe stopped (expected):', (err as Error)?.message)
        return
      }
      log('pipe FAILED:', err)
      emit('error', {
        type:    'worker-crash',
        message: `pipe failed: ${(err as Error)?.message ?? String(err)}`,
        source:  'worker',
      })
    })
}

// input → sink, one frame at a time. Instead of pipeTo() because every frame
// must be closed whatever happens: pipeTo drops a chunk it has read but not
// written when it is aborted or the sink errors (GPU lost mid-frame), and
// camera frames carry the page's close auditor, so a frame dropped here is
// reported in the app's console ("A VideoFrame was garbage collected without
// being closed") and pins a capture buffer until GC. Leaves the sink open and
// unlocked (camera switch reuses it); cancels the input when it stops.
async function pump(input: ReadableStream<VideoFrame>, sink: WritableStream<VideoFrame>, signal: AbortSignal): Promise<void> {
  const reader = input.getReader()
  const writer = sink.getWriter()
  const onAbort = () => { reader.cancel().catch(() => {}) }
  signal.addEventListener('abort', onAbort)
  let ended = false
  try {
    for (;;) {
      await writer.ready
      if (signal.aborted) return
      const { value, done } = await reader.read()
      if (done) { ended = true; return }
      if (signal.aborted) { value.close(); return }
      try {
        await writer.write(value)
      } catch (err) {
        value.close()   // no-op if the sink already closed it
        throw err
      }
    }
  } finally {
    signal.removeEventListener('abort', onAbort)
    if (!ended) await reader.cancel().catch(() => {})
    reader.releaseLock()
    writer.releaseLock()
  }
}

// Release an input this worker won't pump. Chromium: a transferred
// MediaStreamTrackProcessor readable that is cancelled (or whose worker dies)
// without ever having been read leaks a VideoFrame, reported in the page's
// console. Reading one frame first (closed at once) avoids it.
async function discardInput(input: ReadableStream<VideoFrame> | undefined): Promise<void> {
  if (!input || input.locked) return
  const reader = input.getReader()
  const first = reader.read().then((r) => { r.value?.close() }, () => {})
  await Promise.race([first, new Promise((r) => setTimeout(r, 200))])
  await reader.cancel().catch(() => {})
}

function stopInputPipe(): void {
  pipeGen++
  inputAbort?.abort()
  inputAbort = null
}

async function handleInit(data: InitData): Promise<InitResponse> {
  // Honor the consumer's debug flag inside the worker (its own context).
  // Set this BEFORE the first log() of this call so the message respects it.
  setDebug(!!data.debug)
  log('handleInit: start; topology=', data.topology, 'preset=', data.preset, 'backend=', data.backend, 'dtype=', data.dtype)
  if (initData) throw new Error('handleInit: already initialized')
  initData = data
  startHeartbeat(data.heartbeatMs ?? 1000)

  // dtype: f16 if backend supports shader-f16, else f32. Preset's intended
  // dtype (e.g. PRESETS lists large/xl as f32 for accuracy) is only a
  // spec/intent, not enforced — runtime always uses the cheaper f16 when
  // it's available. Trade-off: simpler than rebuilding the backend per-
  // preset, but bench timings underestimate the cost of large/xl on
  // devices that would otherwise need f32. Acceptable for v0.1.
  log('handleInit: setupBackend…')
  let setup: Awaited<ReturnType<typeof setupBackend>> | null = null
  let backendError: string | undefined
  if (data.backend === 'none') {
    backendError = 'GPU backend disabled'
  } else {
    try {
      setup = await setupBackend(data, data.canvasSize, {
        onContextLost: (err) => {
          if (backendLost) return
          backendLost = true
          log('GPU context lost:', err.message)
          // Stop feeding a dead context; main switches the output to raw and
          // restarts this worker with a fresh backend.
          stopInputPipe()
          emit('error', err)
        },
        isShuttingDown: () => shuttingDown,
      })
    } catch (e) {
      backendError = (e as Error)?.message ?? String(e)
    }
  }

  // Destroyed while the backend was being set up: handleDestroy released
  // the input; don't build a renderer on a worker that is about to go.
  if (shuttingDown) {
    setup?.backend.destroy()
    throw new Error('handleInit: worker destroyed during init')
  }

  const input = createInputStream(data)

  if (!setup) {
    // ── No GPU: fail open with raw frames ────────────────────────────────
    noneMode = true
    log('handleInit: no backend —', backendError)
    const rawSink = createRawOutputSink(data, onFrame)
    if (rawSink) {
      outputSink = rawSink
      startInputPipe(input, rawSink)
    } else {
      // Main side provides raw (relay / main canvas). Release our input.
      void discardInput(input)
    }
    resolvedPreset = null
    return {
      resolvedPreset:  null,
      resolvedBackend: 'none',
      resolvedDtype:   data.dtype,
      workerDrawsRaw:  !!rawSink,
      backendError,
    }
  }
  log('handleInit: backend ready:', setup.resolvedBackend, setup.resolvedDtype, 'canvas:', setup.canvas.width, 'x', setup.canvas.height)

  // Construct Renderer in boot mode (no preset, no weights, no network).
  // Pipe starts immediately so passthrough is flowing while autotune runs.
  log('handleInit: constructing Renderer (boot mode)…')
  renderer = new Renderer({
    backend:     setup.backend,
    backendKind: setup.resolvedBackend,
    canvas:      setup.canvas,
    background:  data.background,
    enabled:     data.enabled,
    topology:    data.topology,
  })

  log('handleInit: createOutputSink…')
  outputSink = createOutputSink(data, renderer, onFrame)
  log('handleInit: starting pipe (passthrough)…')
  startInputPipe(input, outputSink)

  // Autotune (or named-preset lookup). Pipe is already pumping passthrough,
  // so the consumer sees live video while this runs (~1-2s on cold start).
  // Note: autotune competes with passthrough for GPU; passthrough is cheap
  // (one Input.run + one CompositePassthrough) so contention is minor.
  model = data.model ?? { family: 'longpipe' }
  if (model.family === 'mediapipe-selfie') {
    // One fixed model — nothing to autotune. startRender carries its weights.
    log('handleInit: MediaPipe selfie', model.variant, '— skipping autotune')
    resolvedPreset = null
    return { resolvedPreset: null, resolvedBackend: setup.resolvedBackend, resolvedDtype: setup.resolvedDtype }
  }

  log('handleInit: resolvePreset…')
  const preset = await resolvePreset(data.preset, setup.resolvedDtype, setup.backend)
  log('handleInit: preset resolved:', preset)
  resolvedPreset = preset

  log('handleInit: done; returning InitResponse, awaiting startRender')
  return {
    resolvedPreset:  preset,
    resolvedBackend: setup.resolvedBackend,
    resolvedDtype:   setup.resolvedDtype,
  }
}

async function handleStartRender({ weights, preset }: CmdDataMap['startRender']): Promise<void> {
  log('handleStartRender: start; weights bytes:', weights.byteLength)
  if (preset) resolvedPreset = preset
  if (!renderer) throw new Error('handleStartRender: handleInit not completed')

  if (model.family === 'mediapipe-selfie') {
    log('handleStartRender: attaching MediaPipe selfie', model.variant)
    renderer.setMediaPipeSelfie(model.variant, weights)
  } else {
    if (!resolvedPreset) throw new Error('handleStartRender: handleInit not completed')
    log('handleStartRender: attaching network via setPreset…')
    renderer.setPreset(resolvedPreset, weights)
  }

  log('handleStartRender: ready (effect mode active)')
  emit('ready', undefined)
}

function requireRenderer(cmd: string): Renderer {
  if (!renderer) throw new Error(noneMode ? `${cmd}: no GPU backend` : `${cmd}: not initialized`)
  return renderer
}

async function handleSetBackground(bg: Background): Promise<void> {
  requireRenderer('setBackground').setBackground(bg)
}

async function handleSetEnabled(on: boolean): Promise<void> {
  renderer?.setEnabled(on)
}

async function handleAttachPreview(data: CmdDataMap['attachPreview']): Promise<void> {
  requireRenderer('attachPreview').attachPreview(data.canvas)
}

async function handleSetPreview(data: CmdDataMap['setPreview']): Promise<void> {
  requireRenderer('setPreview').setPreview(data.background, data.fps !== undefined ? { fps: data.fps } : undefined)
}

async function handleClearPreview(): Promise<void> {
  renderer?.clearPreview()
}

async function handleSetPreset(
  data: CmdDataMap['setPreset'],
): Promise<PresetSwapResult> {
  const r = requireRenderer('setPreset')
  if (model.family === 'mediapipe-selfie')
    throw new Error("handleSetPreset: presets select Longpipe tiers; not available with a MediaPipe model")
  if (!data.weights)  throw new Error('handleSetPreset: weights required for runtime preset swap')
  if (data.preset === 'auto') {
    throw new Error("handleSetPreset: 'auto' not supported on runtime swap (use at init for autotune)")
  }

  // Caller is responsible for supplying weights compatible with the
  // current backend's dtype. We don't re-probe here.
  const resolved: ManualPreset = typeof data.preset === 'string'
    ? (resolveNamedPreset(data.preset) ?? throwUnknown(data.preset))
    : data.preset

  r.setPreset(resolved, data.weights)
  resolvedPreset = resolved
  return { resolvedPreset: resolved }
}

// Camera switch: pipe a new input into the same output sink.
async function handleSetInput(data: CmdDataMap['setInput']): Promise<void> {
  if (!initData) throw new Error('setInput: not initialized')
  const next: InitData = { ...initData, inputReadable: data.inputReadable, inputPort: data.inputPort }
  const input = createInputStream(next)
  initData = next
  if (!outputSink) {
    // No-GPU on a main-raw path: nothing reads input here.
    void discardInput(input)
    return
  }
  const prev = pipeDone
  stopInputPipe()
  // Wait for the old pipe to release the sink's lock before piping again.
  await prev?.catch(() => {})
  startInputPipe(input, outputSink)
}

function getStats(): RendererStats | null {
  if (renderer) return { ...renderer.getStats(), latencyMs: meanLatencyMs() }
  if (!initData) return null
  const now = performance.now()
  while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift()
  return {
    fps: frameTimes.length, modelFps: 0, modelMs: 0, displayMs: 0, skipped: 0,
    preset: 'none', skipFrames: 0, enabled: false,
    inputPath: initData.topology.input, outputPath: initData.topology.output,
    backend: 'none', latencyMs: meanLatencyMs(),
  }
}

async function handleDebug(data: CmdDataMap['debug']): Promise<void> {
  switch (data.action) {
    case 'lose-backend': {
      const b = requireRenderer('debug').backendHandle as unknown as {
        device?: GPUDevice
        gl?: WebGL2RenderingContext
      }
      if (b.device) b.device.destroy()
      else if (b.gl) {
        const ext = b.gl.getExtension('WEBGL_lose_context')
        if (!ext) throw new Error('WEBGL_lose_context unavailable')
        ext.loseContext()
      } else throw new Error('debug: unknown backend')
      return
    }
    case 'hang':  return new Promise<void>(() => { /* never settles */ })
    case 'throw': throw new Error('debug: simulated failure')
  }
}

async function handleDestroy(): Promise<void> {
  // Order matters: abort the pipe first so no more frames flow into the
  // renderer, THEN destroy the renderer (releases backend / GPU).
  // Reversed order can fire process() on a destroyed backend.
  shuttingDown = true
  if (heartbeat) clearInterval(heartbeat)
  heartbeat = null
  // Let the frame being written finish (its sink closes it) and the input be
  // cancelled (queued frames closed) before the backend goes away; main
  // terminates this worker once we answer.
  const pipe = pipeDone
  stopInputPipe()
  await Promise.race([
    Promise.all([
      pipe?.catch(() => {}),
      // Destroyed before init got to start the pipe (e.g. while the backend
      // was still being set up): the camera readable was never read.
      discardInput(initData?.inputReadable),
    ]),
    new Promise((r) => setTimeout(r, 300)),
  ])
  renderer?.destroy()
  renderer = null
  resolvedPreset = null
}

function throwUnknown(name: PresetName): never {
  throw new Error(`handleSetPreset: unknown preset '${name}'`)
}

function respond(request_id: string, res: unknown): void {
  const reply: WorkerResponse = { request_id, res } as WorkerResponse
  ;(self as unknown as Worker).postMessage(reply)
}

function respondError(request_id: string, message: string): void {
  const reply: WorkerResponse = { request_id, error: { message } }
  ;(self as unknown as Worker).postMessage(reply)
}

function emit<E extends EventName>(name: E, res: EventMap[E]): void {
  const event: WorkerEvent<E> = { request_id: name, res }
  try {
    ;(self as unknown as Worker).postMessage(event)
  } catch {
    // Uncloneable payload (e.g. a `cause` object) — send without it.
    const r = res as { cause?: unknown }
    if (r && typeof r === 'object' && 'cause' in r) {
      ;(self as unknown as Worker).postMessage({ request_id: name, res: { ...r, cause: undefined } })
    }
  }
}

// Convenience for renderer to push stats up (wired via a callback in
// handleInit when we add periodic stats reporting).
export function emitStats(stats: RendererStats): void {
  emit('stats', stats)
}

// Face touch-up: rebuild the renderer's face-effect chain from the payload
// (or clear it). The renderer defers until a preset is attached, so this is
// safe at any point after init.
function handleSetReframe(cfg: import('../messages').ReframeConfig | null): void {
  requireRenderer('setReframe').setReframe(cfg)
}

function handleReframeNow(): void {
  requireRenderer('reframeNow').reframeNow()
}

function handleSetTouchup(p: import('../messages').TouchupPayload | null): void {
  requireRenderer('setTouchup').setFaceEffects(p === null ? null : {
    landmarkWeights: p.landmarkWeights,
    topology: { count: p.topoCount, uv: p.topoUv, idx: p.topoIdx, weightMask: p.weightMask },
    touchup: p.params,
  })
}
