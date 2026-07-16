import { describe, it, expect } from 'vitest'
import { Renderer, type FaceEffectsConfig } from '~/pipeline/worker/renderer'
import type { Backend, FaceTopology } from '~/model/backend'
import { createWebGPUBackend, createWebGLBackend } from '../helpers/backends'
import topoJson from '../fixtures/face_topology.json'

// Renderer-level smoke of the hydranet path: setPreset (real small .bin with
// face+flow) + setFaceEffects (real landmark .bin) → process() across
// inference AND skip frames. Exercises stepFlow + stepFace (heatmap warp on
// skips) + the effect chain + fg-passthrough / effect composites on both
// backends. Synthetic frames carry no face (box score ≈ 0), so the touch-up
// stage passes through — the test asserts the plumbing runs clean, not the
// visual effect (the demos/probe pages cover that).
//
// LOCAL fixtures (gitignored): self-skips on a clean checkout.
const binUrls = (import.meta as any).glob('../fixtures/local/*.bin',
  { query: '?url', import: 'default', eager: true }) as Record<string, string>
const tierUrl = Object.entries(binUrls).find(([p]) => p.endsWith('model_small.bin'))?.[1]
const lmUrl   = Object.entries(binUrls).find(([p]) => p.endsWith('model_landmark_mesh.bin'))?.[1]

const BACKENDS: Array<{ name: string; kind: 'webgpu' | 'webgl'; create: (c: OffscreenCanvas) => Promise<Backend> }> = [
  { name: 'WebGPU', kind: 'webgpu', create: () => createWebGPUBackend() },
  { name: 'WebGL',  kind: 'webgl',  create: async () => createWebGLBackend() },
]

async function makeTopo(): Promise<FaceTopology> {
  const c = new OffscreenCanvas(512, 512)
  const ctx = c.getContext('2d')!
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, 512, 512)
  return {
    count: (topoJson as any).count,
    uv: new Float32Array((topoJson as any).uv),
    idx: new Float32Array((topoJson as any).idx),
    weightMask: await createImageBitmap(c),
  }
}

function makeFrame(t: number): VideoFrame {
  const c = new OffscreenCanvas(384, 224)
  const ctx = c.getContext('2d')!
  ctx.fillStyle = `rgb(${(t * 40) % 255}, 120, 80)`
  ctx.fillRect(0, 0, 384, 224)
  ctx.fillStyle = '#222'
  ctx.fillRect(40 + t * 5, 30, 120, 140)
  return new VideoFrame(c, { timestamp: t * 33_000 })
}

describe.each(BACKENDS)('Renderer face chain ($name)', ({ name, kind, create }) => {
  it.skipIf(!tierUrl || !lmUrl)('runs inference + skip frames with the face chain active', async () => {
    // WebGPUBackend.create in the helper builds its own canvas; the renderer
    // needs the SAME canvas — create the backend around a known OffscreenCanvas.
    const canvas = new OffscreenCanvas(384, 224)
    const backend = kind === 'webgpu'
      ? await (await import('~/model/backends/webgpu/index')).WebGPUBackend.create({ canvas, dtype: 'f32' })
      : (await import('~/model/backends/webgl/index')).WebGLBackend.create({ canvas, dtype: 'f32' })

    const renderer = new Renderer({
      backend, backendKind: kind, canvas,
      background: { kind: 'color', rgb: [0, 1, 0] },
      enabled: true,
      topology: { input: 'postmessage', output: 'bitmap-shuttle' } as any,
    })

    const [tierBuf, lmBuf] = await Promise.all([
      fetch(tierUrl!).then(r => r.arrayBuffer()),
      fetch(lmUrl!).then(r => r.arrayBuffer()),
    ])
    renderer.setPreset({ model: 'small', dtype: 'f32', resolution: { w: 384, h: 224 }, skipFrames: 2 }, tierBuf)

    const cfg: FaceEffectsConfig = {
      landmarkWeights: lmBuf,
      topology: await makeTopo(),
      touchup: { strength: 0.7, amount: 8, detail: 0.35, thresh: 0.15 },
    }
    renderer.setFaceEffects(cfg)

    // 7 frames: covers inference frames (0, 3, 6) and skip frames (warp path).
    for (let t = 0; t < 7; t++) {
      const f = makeFrame(t)
      renderer.process(f)
      f.close()
    }
    await backend.sync()

    // Background 'none' + face chain → fg-passthrough branch.
    renderer.setBackground({ kind: 'none' })
    for (let t = 7; t < 10; t++) {
      const f = makeFrame(t)
      renderer.process(f)
      f.close()
    }
    await backend.sync()

    const stats = renderer.getStats()
    expect(stats.preset).toBe('small')
    expect(stats.fps).toBeGreaterThan(0)
    renderer.destroy()
  })
})
