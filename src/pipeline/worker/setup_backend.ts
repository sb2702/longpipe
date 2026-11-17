// Constructs the GPU backend in the worker, honoring the caller's
// preference (`'webgpu' | 'webgl' | 'auto'`) but falling back gracefully
// when the preferred path isn't available.
//
// Decision flow:
//   1. If preference is 'webgpu' or 'auto' AND WebGPUBackend.isAvailable():
//      a. If dtype === 'f16' but no shader-f16, downgrade to f32 silently
//      b. Try WebGPUBackend.create(); on success, return
//      c. On failure (probe lied / driver quirk), log + fall through to WebGL
//   2. Try WebGLBackend.isAvailable() + create(); return on success
//   3. Throw if neither works
//
// Caveat — single-canvas attempts: when output topology is
// 'transfer-capture', the canvas is the same one main side will captureStream
// from. Once a context type (webgpu vs webgl2) has been attached to that
// canvas, the other type can no longer use it. So if WebGPU partially
// succeeds in attaching a context and then throws, the WebGL fallback on
// that same canvas will probably also throw. For non-transfer-capture
// topologies the worker owns a fresh OffscreenCanvas per attempt.

import type { Backend, Dtype } from '~/model/backend.ts'
import { WebGPUBackend } from '~/model/backends/webgpu/index.ts'
import { WebGLBackend }  from '~/model/backends/webgl/index.ts'
import type { InitData, WorkerErrorEvent } from '../messages'
import { createLogger } from '../debug'

const log = createLogger('setup_backend')

export interface BackendSetup {
  backend:         Backend
  resolvedBackend: 'webgpu' | 'webgl'
  resolvedDtype:   Dtype
  canvas:          OffscreenCanvas
}

export interface SetupBackendOptions {
  // Fires once if the GPU backend dies after a successful setup —
  // WebGPU device.lost or WebGL webglcontextlost. The renderer is unusable
  // after this; the worker stops its pipe and main restarts it (raw
  // passthrough meanwhile).
  onContextLost?: (err: WorkerErrorEvent) => void
  // True while the worker is tearing down on purpose. Our own destroy()
  // resolves device.lost with reason 'destroyed'; anything else (including
  // a 'destroyed' we didn't ask for) is a real loss.
  isShuttingDown?: () => boolean
}

export async function setupBackend(
  data:        InitData,
  canvasSize:  { w: number; h: number },
  opts:        SetupBackendOptions = {},
): Promise<BackendSetup> {
  const wantWebGPU = data.backend === 'webgpu' || data.backend === 'auto'
  const isTransferCapture = !!data.outputCanvas

  // For transfer-capture, we MUST use the canvas main side gave us (it's
  // what captureStream observes). Otherwise allocate fresh per attempt so
  // a failed attempt's partial context state doesn't poison the next try.
  const canvasFor = (): OffscreenCanvas =>
    data.outputCanvas ?? new OffscreenCanvas(canvasSize.w, canvasSize.h)

  let lastError: unknown = null

  if (wantWebGPU && await WebGPUBackend.isAvailable()) {
    const dtype: Dtype = (data.dtype === 'f16' && !(await WebGPUBackend.hasF16Support()))
      ? 'f32'
      : data.dtype
    const canvas = canvasFor()
    try {
      const backend = await WebGPUBackend.create({ canvas, dtype })
      // device.lost resolves on driver crash, GPU reset, or destroy().
      // We only surface if NOT triggered by our own destroy() — the
      // 'destroyed' reason indicates intentional teardown.
      backend.device.lost.then(info => {
        if (opts.isShuttingDown?.()) return
        opts.onContextLost?.({
          type:    'backend-lost',
          message: `WebGPU device lost: ${info.reason}${info.message ? ' — ' + info.message : ''}`,
          source:  'backend-lost',
          cause:   { reason: info.reason, message: info.message },
        })
      }).catch(() => { /* device.lost itself rejecting is unexpected; ignore */ })
      return { backend, resolvedBackend: 'webgpu', resolvedDtype: dtype, canvas }
    } catch (e) {
      log.warn('WebGPU isAvailable but create() threw; trying WebGL fallback:', e)
      lastError = e
      if (isTransferCapture) {
        log.warn('transfer-capture topology: WebGL fallback may also fail on the same canvas')
      }
    }
  }

  if (WebGLBackend.isAvailable()) {
    const canvas = canvasFor()
    try {
      const backend = WebGLBackend.create({ canvas, dtype: data.dtype })
      // OffscreenCanvas extends EventTarget — webglcontextlost fires here
      // when the context is lost. Recovery is a fresh worker + backend
      // (driven by main), so we don't wait for webglcontextrestored.
      canvas.addEventListener('webglcontextlost', () => {
        if (opts.isShuttingDown?.()) return
        opts.onContextLost?.({
          type:    'backend-lost',
          message: 'WebGL context lost',
          source:  'backend-lost',
        })
      })
      return { backend, resolvedBackend: 'webgl', resolvedDtype: data.dtype, canvas }
    } catch (e) {
      log.warn('WebGL create() threw:', e)
      lastError = e
    }
  }

  const detail = lastError ? ` (last error: ${(lastError as Error).message ?? lastError})` : ''
  throw new Error(`setup_backend: no usable GPU backend${detail}`)
}
