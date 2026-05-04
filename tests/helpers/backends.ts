import type { Backend, Dtype } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend  } from '~/model/backends/webgl/index'

// Fresh OffscreenCanvas per backend instance — works on the main thread and
// inside a worker (per project_backend_canvas_contract.md). Tests don't care
// about the canvas dimensions; they read tensor outputs back, not pixels.
const freshCanvas = (): OffscreenCanvas => new OffscreenCanvas(1, 1)

export const createWebGPUBackend = (dtype: Dtype = 'f32'): Promise<WebGPUBackend> =>
  WebGPUBackend.create({ canvas: freshCanvas(), dtype })

export const createWebGLBackend = (dtype: Dtype = 'f32'): WebGLBackend =>
  WebGLBackend.create({ canvas: freshCanvas(), dtype })

// For dual-backend block / network tests using describe.each.
export const BACKENDS: Array<{ name: string; create: () => Promise<Backend> }> = [
  { name: 'WebGPU', create: () => createWebGPUBackend() },
  { name: 'WebGL',  create: async () => createWebGLBackend() },
]

// Dual-backend × dual-precision matrix for tests that exercise fp16. Fp16 is
// optional on WebGPU (requires `shader-f16`); the matrix entry resolves at
// test time and any fp16 row that can't be created should be skipped by the
// test itself.
export const BACKENDS_DTYPE: Array<{ name: string; dtype: Dtype; create: () => Promise<Backend> }> = [
  { name: 'WebGPU-f32', dtype: 'f32', create: () => createWebGPUBackend('f32') },
  { name: 'WebGPU-f16', dtype: 'f16', create: () => createWebGPUBackend('f16') },
  { name: 'WebGL-f32',  dtype: 'f32', create: async () => createWebGLBackend('f32') },
  { name: 'WebGL-f16',  dtype: 'f16', create: async () => createWebGLBackend('f16') },
]
