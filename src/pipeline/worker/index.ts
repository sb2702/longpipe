// Worker entry point. Two responsibilities:
//
//   1. Dispatch incoming control-plane messages (init / setBackground /
//      setEnabled / setPreset / getStats / destroy) to typed handlers.
//   2. handleInit: setupBackend → construct Renderer in boot mode →
//      wire input + output + start pipe (passthrough flowing immediately)
//      → resolvePreset (autotune; pipe keeps pumping passthrough during
//      this) → return InitResponse.
//   3. handleStartRender: renderer.setPreset(preset, weights) — attaches
//      the network in place. Renderer flips from passthrough to effect
//      mid-pipe; no pipe restart, no stream relock.
//
// All actual work lives in the submodules; this file should stay thin.

const log = (...args: unknown[]) => console.log('[longpipe/worker]', ...args)
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
import { createOutputSink }  from './create_output'
import { startPipe }         from './pipe'

let renderer:  Renderer        | null = null
let pumpAbort: AbortController | null = null

// State held between handleInit and handleStartRender — the latter only
// needs to know which preset the autotune chose (so it can pass it to
// renderer.setPreset alongside the weights main fetched).
let resolvedPreset: ManualPreset | null = null

// Default canvas size for non-transfer-capture topologies. For transfer-
// capture, main supplies the canvas and this is unused. TODO: let main
// pass a desired output resolution via InitData; 720p is a reasonable
// default for video-call output but power users will want to override.
const DEFAULT_CANVAS = { w: 1280, h: 720 }

self.onmessage = async function (event: MessageEvent<WorkerRequest>) {
  const { cmd, data, request_id } = event.data
  log('cmd received:', cmd, 'request_id:', request_id)
  try {
    const res = await handleCommand(cmd, data)
    log('cmd ok:', cmd)
    respond(request_id, res)
  } catch (err) {
    log('cmd FAILED:', cmd, err)
    emit('error', { message: (err as Error).message ?? String(err), recoverable: false })
  }
}

async function handleCommand(cmd: CmdName, data: unknown): Promise<unknown> {
  switch (cmd) {
    case 'init':          return handleInit(data as InitData)
    case 'startRender':   return handleStartRender((data as CmdDataMap['startRender']).weights)
    case 'setBackground': return handleSetBackground(data as Background)
    case 'setEnabled':    return handleSetEnabled((data as CmdDataMap['setEnabled']).enabled)
    case 'setPreset':   return handleSetPreset(data as CmdDataMap['setPreset'])
    case 'getStats':    return renderer?.getStats() ?? null
    case 'destroy':     return handleDestroy()
    default: throw new Error(`unknown cmd: ${cmd}`)
  }
}

async function handleInit(data: InitData): Promise<InitResponse> {
  log('handleInit: start; topology=', data.topology, 'preset=', data.preset, 'backend=', data.backend, 'dtype=', data.dtype)
  if (renderer) throw new Error('handleInit: already initialized')

  // dtype: f16 if backend supports shader-f16, else f32. Preset's intended
  // dtype (e.g. PRESETS lists large/xl as f32 for accuracy) is only a
  // spec/intent, not enforced — runtime always uses the cheaper f16 when
  // it's available. Trade-off: simpler than rebuilding the backend per-
  // preset, but bench timings underestimate the cost of large/xl on
  // devices that would otherwise need f32. Acceptable for v0.1.
  log('handleInit: setupBackend…')
  const setup = await setupBackend(data, DEFAULT_CANVAS)
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

  pumpAbort = new AbortController()
  log('handleInit: createInputStream…')
  const input  = createInputStream(data)
  log('handleInit: createOutputSink…')
  const output = createOutputSink(data, renderer, pumpAbort.signal)

  log('handleInit: starting pipe (passthrough)…')
  startPipe({
    input,
    output,
    signal: pumpAbort.signal,
  }).then(() => {
    log('pipe completed (input ended)')
  }).catch(err => {
    if (pumpAbort?.signal.aborted) {
      log('pipe aborted (expected on destroy)')
      return
    }
    log('pipe FAILED:', err)
    emit('error', { message: `pipe failed: ${(err as Error).message}`, recoverable: false })
  })

  // Autotune (or named-preset lookup). Pipe is already pumping passthrough,
  // so the consumer sees live video while this runs (~1-2s on cold start).
  // Note: autotune competes with passthrough for GPU; passthrough is cheap
  // (one Input.run + one CompositePassthrough) so contention is minor.
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

async function handleStartRender(weights: ArrayBuffer): Promise<void> {
  log('handleStartRender: start; weights bytes:', weights.byteLength)
  if (!renderer || !resolvedPreset) throw new Error('handleStartRender: handleInit not completed')

  log('handleStartRender: attaching network via setPreset…')
  renderer.setPreset(resolvedPreset, weights)

  log('handleStartRender: ready (effect mode active)')
  emit('ready', undefined)
}

async function handleSetBackground(bg: Background): Promise<void> {
  renderer?.setBackground(bg)
}

async function handleSetEnabled(on: boolean): Promise<void> {
  renderer?.setEnabled(on)
}

async function handleSetPreset(
  data: CmdDataMap['setPreset'],
): Promise<PresetSwapResult> {
  if (!renderer)      throw new Error('handleSetPreset: not initialized')
  if (!data.weights)  throw new Error('handleSetPreset: weights required for runtime preset swap')
  if (data.preset === 'auto') {
    throw new Error("handleSetPreset: 'auto' not supported on runtime swap (use at init for autotune)")
  }

  // Caller is responsible for supplying weights compatible with the
  // current backend's dtype. We don't re-probe here.
  const resolved: ManualPreset = typeof data.preset === 'string'
    ? (resolveNamedPreset(data.preset) ?? throwUnknown(data.preset))
    : data.preset

  renderer.setPreset(resolved, data.weights)
  return { resolvedPreset: resolved }
}

async function handleDestroy(): Promise<void> {
  pumpAbort?.abort()
  pumpAbort = null
  // TODO: backend.destroy() to release GPU resources (no destroy() on
  // Backend interface yet — add when we wire actual teardown).
  renderer = null
}

function throwUnknown(name: PresetName): never {
  throw new Error(`handleSetPreset: unknown preset '${name}'`)
}

function respond(request_id: string, res: unknown): void {
  const reply: WorkerResponse = { request_id, res } as WorkerResponse
  ;(self as unknown as Worker).postMessage(reply)
}

function emit<E extends EventName>(name: E, res: EventMap[E]): void {
  const event: WorkerEvent<E> = { request_id: name, res }
  ;(self as unknown as Worker).postMessage(event)
}

// Convenience for renderer to push stats up (wired via a callback in
// handleInit when we add periodic stats reporting).
export function emitStats(stats: RendererStats): void {
  emit('stats', stats)
}
