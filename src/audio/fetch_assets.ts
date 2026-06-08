// Fetch + compile a denoise kernel's assets on the MAIN thread (the worklet
// scope can't fetch). Mirrors the video side's fetchModelWeights. The compiled
// WebAssembly.Module and the weights ArrayBuffer are then handed into the
// processor via processorOptions (Module is structured-cloneable).

import { KERNELS, type DenoiseModel } from './kernels'

export interface KernelAssets {
  wasmBytes: ArrayBuffer        // compiled in the worklet, not here (see ProcessorInit)
  weights:   ArrayBuffer | null
}

function urlFor(baseUrl: string, file: string): string {
  return `${baseUrl.replace(/\/$/, '')}/${file}`
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`audio asset fetch failed: ${r.status} ${url}`)
  return r.arrayBuffer()
}

// Fetch the wasm (compile to a Module) and the weights pack (if the model needs
// one — rnnoise embeds its weights). Runs the two fetches concurrently.
export async function fetchKernel(model: DenoiseModel, baseUrl: string): Promise<KernelAssets> {
  const spec = KERNELS[model]
  const [wasmBytes, weights] = await Promise.all([
    fetchBuffer(urlFor(baseUrl, spec.wasm)),
    spec.weights ? fetchBuffer(urlFor(baseUrl, spec.weights)) : Promise.resolve(null),
  ])
  return { wasmBytes, weights }
}

// wasm-SIMD capability probe (DFN kernels require simd128; rnnoise has a scalar
// fallback). Used by tier selection to gate DFN out on no-SIMD devices.
export function simdSupported(): boolean {
  try {
    // Minimal `() -> v128` module (a v128.const). v128.const takes 16 immediate
    // bytes — getting that count wrong makes validate() always return false.
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
      10, 22, 1, 20, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
    ]))
  } catch { return false }
}
