// Weight-free tier probe for `model: 'auto'`. Picks high (dfn) / mid (dfnint8) /
// low (rnnoise) by timing the REAL net against synthetic weights on THIS device
// — no multi-MB weights download (per-hop cost is shape-driven, not value-driven;
// see audio-denoising HANDOFF §9). Runs ONLY for 'auto'; an explicit model or
// tier skips it entirely.
//
// The timing runs in a throwaway Worker, NOT the main thread: calibrate is a
// blocking loop of net runs (~tens of ms on fast HW, more on slow — and the
// probe matters most on slow HW), so blocking the main thread there is exactly
// backwards. The worker fetches/compiles dfn.wasm, times the calibrate calls,
// posts back the chosen model, and is terminated.

import { simdSupported } from './fetch_assets'
import { KERNELS, type DenoiseModel } from './kernels'

// Per-hop budget. The hop runs synchronously inside one process() call against
// the render-quantum deadline (128/48000 ≈ 2.67 ms @ 48 kHz). Require ~2×
// headroom — and startup is the device's coolest/least-loaded moment, so real
// sustained cost is worse; the budget is deliberately well under the deadline.
const BUDGET_MS = 1.3
const ITERS = 64    // net runs per timed call
const ROUNDS = 3    // best-of, to reject scheduler noise

// Self-contained worker (no imports → Blob-URL spawn, same pattern as the main
// SDK worker, works in dev + published builds). Parameterized via the message.
const WORKER_SRC = `
self.onmessage = async (e) => {
  const { wasmUrl, iters, rounds, budget } = e.data
  try {
    const res = await fetch(wasmUrl)
    if (!res.ok) throw new Error(res.status + ' ' + wasmUrl)
    const module = await WebAssembly.compile(await res.arrayBuffer())
    const imports = {}
    for (const im of WebAssembly.Module.imports(module)) {
      ;(imports[im.module] = imports[im.module] || {})
      if (im.kind === 'function') imports[im.module][im.name] = () => 0
    }
    const ex = new WebAssembly.Instance(module, imports).exports
    const best = (fn) => {
      fn(iters)                                   // warm up
      let b = Infinity
      for (let r = 0; r < rounds; r++) {
        const t = performance.now()
        fn(iters)
        b = Math.min(b, (performance.now() - t) / iters)
      }
      return b
    }
    const dfnMs = best(ex.calibrate_f32)
    const int8Ms = ex.calibrate_i8 ? best(ex.calibrate_i8) : Infinity
    const model = dfnMs <= budget ? 'dfn' : int8Ms <= budget ? 'dfnint8' : 'rnnoise'
    self.postMessage({ model, dfnMs, int8Ms })
  } catch (err) {
    self.postMessage({ error: (err && err.message) || String(err) })
  }
}
`

// Resolve 'auto' → a concrete model. The worker fetches only dfn.wasm (~hundreds
// of KB, and the same file the chosen DFN tier reuses from the HTTP cache); never
// the weight packs. On any failure, falls back to the rnnoise floor.
export async function selectTier(baseUrl: string, onError?: (m: string) => void): Promise<DenoiseModel> {
  // DFN needs SIMD (the wasm won't even instantiate without it) — gate here,
  // instantly, before spawning a worker.
  if (!simdSupported()) return 'rnnoise'

  const wasmUrl = `${baseUrl.replace(/\/$/, '')}/${KERNELS.dfn.wasm}`
  let url: string | null = null
  let worker: Worker | null = null
  try {
    url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'application/javascript' }))
    worker = new Worker(url)
    const result = await new Promise<{ model?: DenoiseModel; error?: string }>((resolve, reject) => {
      worker!.onmessage = (e: MessageEvent) => resolve(e.data)
      worker!.onerror = (e) => reject(new Error(e.message || 'probe worker error'))
      worker!.postMessage({ wasmUrl, iters: ITERS, rounds: ROUNDS, budget: BUDGET_MS })
    })
    if (result.error) throw new Error(result.error)
    return result.model ?? 'rnnoise'
  } catch (e) {
    onError?.(`audio tier probe failed (${(e as Error).message ?? String(e)}); using rnnoise`)
    return 'rnnoise'
  } finally {
    worker?.terminate()
    if (url) URL.revokeObjectURL(url)
  }
}
