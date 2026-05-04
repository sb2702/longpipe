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

import type { Backend, Dtype } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu'
import { WebGLBackend }  from '~/model/backends/webgl'
import type { InitData } from '../messages'

export interface BackendSetup {
  backend:         Backend
  resolvedBackend: 'webgpu' | 'webgl'
  resolvedDtype:   Dtype
  canvas:          OffscreenCanvas
}

export async function setupBackend(
  data:        InitData,
  canvasSize:  { w: number; h: number },
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
      return { backend, resolvedBackend: 'webgpu', resolvedDtype: dtype, canvas }
    } catch (e) {
      console.warn('[setup_backend] WebGPU isAvailable but create() threw; trying WebGL fallback:', e)
      lastError = e
      if (isTransferCapture) {
        console.warn('[setup_backend] transfer-capture topology: WebGL fallback may also fail on the same canvas')
      }
    }
  }

  if (WebGLBackend.isAvailable()) {
    const canvas = canvasFor()
    try {
      const backend = WebGLBackend.create({ canvas, dtype: data.dtype })
      return { backend, resolvedBackend: 'webgl', resolvedDtype: data.dtype, canvas }
    } catch (e) {
      console.warn('[setup_backend] WebGL create() threw:', e)
      lastError = e
    }
  }

  const detail = lastError ? ` (last error: ${(lastError as Error).message ?? lastError})` : ''
  throw new Error(`setup_backend: no usable GPU backend${detail}`)
}
