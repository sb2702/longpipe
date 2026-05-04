import type { Backend } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend  } from '~/model/backends/webgl/index'

// Fresh OffscreenCanvas per backend instance — works on the main thread and
// inside a worker (per project_backend_canvas_contract.md). Tests don't care
// about the canvas dimensions; they read tensor outputs back, not pixels.
const freshCanvas = (): OffscreenCanvas => new OffscreenCanvas(1, 1)

export const createWebGPUBackend = (): Promise<WebGPUBackend> =>
  WebGPUBackend.create({ canvas: freshCanvas() })

export const createWebGLBackend = (): WebGLBackend =>
  WebGLBackend.create({ canvas: freshCanvas() })

// For dual-backend block / network tests using describe.each.
export const BACKENDS: Array<{ name: string; create: () => Promise<Backend> }> = [
  { name: 'WebGPU', create: () => createWebGPUBackend() },
  { name: 'WebGL',  create: async () => createWebGLBackend() },
]
