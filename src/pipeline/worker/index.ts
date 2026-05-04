// Worker entry point — message dispatcher and transport setup.
//
// On 'init': selects the right input/output adapters based on the
// transferred handles, builds the Renderer, wires the pipe, emits 'ready'.
// All other commands forward to the Renderer.

import type {
  WorkerRequest,
  WorkerResponse,
  WorkerEvent,
  CmdName,
  InitData,
  InitResponse,
  EventName,
  EventMap,
  RendererStats,
} from '../messages'
import type { EffectConfig } from '../effects'
import type { ManualPreset, PresetName } from '../presets'
import { Renderer } from './renderer'
import { autotunePreset } from './autotune'
import { resolveNamedPreset } from '../presets'
import { createMstpInput }             from './adapters/input_mstp'
import { createPostMessageInput }      from './adapters/input_postmessage'
import { createMstgOutput }            from './adapters/output_mstg'
import { createTransferCaptureOutput } from './adapters/output_transfer_capture'
import { createBitmapShuttleOutput }   from './adapters/output_bitmap_shuttle'

let renderer:  Renderer        | null = null
let pumpAbort: AbortController | null = null

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
    case 'setEnabled': return handleSetEnabled((data as { enabled: boolean }).enabled)
    case 'setPreset':  return handleSetPreset(data as { preset: PresetName | ManualPreset; weights?: ArrayBuffer })
    case 'getStats':   return renderer?.getStats() ?? null
    case 'destroy':    return handleDestroy()
    default: throw new Error(`unknown cmd: ${cmd}`)
  }
}

async function handleInit(_data: InitData): Promise<InitResponse> {
  // TODO:
  //  1. Construct Backend (WebGPU or WebGL) with:
  //       data.outputCanvas (transfer-capture path)
  //       OR new OffscreenCanvas(preset.resolution.w, preset.resolution.h) (other paths)
  //  2. Resolve preset:
  //       'auto'   → autotunePreset(backend)
  //       named    → resolveNamedPreset()
  //       Manual   → use as-is
  //  3. Construct Renderer with backend/canvas/preset/weights/effect.
  //  4. Build input ReadableStream:
  //       data.inputReadable → createMstpInput
  //       data.inputPort     → createPostMessageInput
  //  5. Build output sink:
  //       data.outputWritable → createMstgOutput → pipeThrough → outputWritable
  //       data.outputCanvas   → createTransferCaptureOutput → pipeTo
  //       data.outputPort     → createBitmapShuttleOutput(port) → pipeTo
  //  6. pumpAbort = new AbortController(); start the pipe; on completion,
  //     emit ready (or error if pipe failed before any frame).
  //  7. Emit 'ready' on first successful render.
  //  8. Return { resolvedPreset }.
  throw new Error('handleInit not yet implemented')
}

async function handleSetEffect(config: EffectConfig): Promise<void> {
  renderer?.setEffect(config)
}

async function handleSetEnabled(on: boolean): Promise<void> {
  renderer?.setEnabled(on)
}

async function handleSetPreset(_data: { preset: PresetName | ManualPreset; weights?: ArrayBuffer }): Promise<{ resolvedPreset: ManualPreset }> {
  // TODO:
  //  - resolve preset (named or manual)
  //  - require data.weights for runtime swap (main side fetches if needed)
  //  - renderer.setPreset(resolved, weights)
  //  - return { resolvedPreset: resolved }
  throw new Error('handleSetPreset not yet implemented')
}

async function handleDestroy(): Promise<void> {
  pumpAbort?.abort()
  pumpAbort = null
  renderer  = null
  // Backend / network teardown happens in destroy chain. TODO: wire up.
}

function respond(request_id: string, res: unknown): void {
  const reply: WorkerResponse = { request_id, res } as WorkerResponse
  ;(self as unknown as Worker).postMessage(reply)
}

function emit<E extends EventName>(name: E, res: EventMap[E]): void {
  const event: WorkerEvent<E> = { request_id: name, res }
  ;(self as unknown as Worker).postMessage(event)
}

// Convenience for renderer to push stats up. Renderer doesn't import this
// directly to keep the abstraction clean; we'll wire a callback in handleInit.
export function emitStats(stats: RendererStats): void {
  emit('stats', stats)
}
