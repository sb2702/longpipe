// Worker entry point. Two responsibilities:
//
//   1. Dispatch incoming control-plane messages (init / setEffect /
//      setEnabled / setPreset / getStats / destroy) to typed handlers.
//   2. handleInit: orchestrate one-time setup by composing the worker
//      submodules — setupBackend, resolvePreset, Renderer, createInputStream,
//      createOutputSink, startPipe.
//
// All actual work lives in the submodules; this file should stay thin.

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
import type { EffectConfig } from '../effects'
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

// Default canvas size for non-transfer-capture topologies. For transfer-
// capture, main supplies the canvas and this is unused. TODO: let main
// pass a desired output resolution via InitData; 720p is a reasonable
// default for video-call output but power users will want to override.
const DEFAULT_CANVAS = { w: 1280, h: 720 }

self.onmessage = async function (event: MessageEvent<WorkerRequest>) {
  const { cmd, data, request_id } = event.data
  try {
    const res = await handleCommand(cmd, data)
    respond(request_id, res)
  } catch (err) {
    emit('error', { message: (err as Error).message ?? String(err), recoverable: false })
  }
}

async function handleCommand(cmd: CmdName, data: unknown): Promise<unknown> {
  switch (cmd) {
    case 'init':       return handleInit(data as InitData)
    case 'setEffect':  return handleSetEffect(data as EffectConfig)
    case 'setEnabled': return handleSetEnabled((data as CmdDataMap['setEnabled']).enabled)
    case 'setPreset':  return handleSetPreset(data as CmdDataMap['setPreset'])
    case 'getStats':   return renderer?.getStats() ?? null
    case 'destroy':    return handleDestroy()
    default: throw new Error(`unknown cmd: ${cmd}`)
  }
}

async function handleInit(data: InitData): Promise<InitResponse> {
  if (renderer) throw new Error('handleInit: already initialized')

  // For non-transfer-capture the canvas is allocated inside setupBackend at
  // this size. For transfer-capture, setupBackend uses data.outputCanvas
  // directly and ignores the size hint.
  const setup = await setupBackend(data, DEFAULT_CANVAS)

  const preset = await resolvePreset(data.preset, setup.resolvedDtype, setup.backend)

  // Weights are required at init time. v0.1 limitation: for preset='auto'
  // the caller can't know in advance which preset's weights to ship, so
  // they should supply weights for a reasonable default (e.g. 'large') or
  // skip 'auto'. Followup: emit 'preset-resolved' after autotune and let
  // main lazy-fetch + send via setPreset before the pipe starts.
  if (!data.weights) {
    throw new Error("handleInit: weights required (v0.1 has no deferred-fetch path for preset='auto')")
  }

  renderer = new Renderer({
    backend: setup.backend,
    canvas:  setup.canvas,
    preset,
    weights: data.weights,
    effect:  data.effect,
    enabled: data.enabled,
  })

  pumpAbort = new AbortController()
  const input  = createInputStream(data)
  const output = createOutputSink(data, renderer, pumpAbort.signal)

  // Start pipe in background — handleInit's response is the init ack, not
  // the end-of-pipe completion. Errors after init come through 'error'
  // events; clean shutdown via destroy() aborts the signal.
  startPipe({
    input,
    output,
    signal: pumpAbort.signal,
    onFirstFrame: () => emit('ready', undefined),
  }).catch(err => {
    if (pumpAbort?.signal.aborted) return        // expected on destroy
    emit('error', { message: `pipe failed: ${(err as Error).message}`, recoverable: false })
  })

  return {
    resolvedPreset:  preset,
    resolvedBackend: setup.resolvedBackend,
    resolvedDtype:   setup.resolvedDtype,
  }
}

async function handleSetEffect(config: EffectConfig): Promise<void> {
  renderer?.setEffect(config)
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
