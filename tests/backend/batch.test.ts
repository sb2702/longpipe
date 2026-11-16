import { describe, it, expect } from 'vitest'
import { createWebGPUBackend } from '../helpers/backends'
import { runBatched } from '~/model/backend'
import type { WebGPUBackend } from '~/model/backends/webgpu/index'

// WebGPU command batching: ops run inside backend.batch() share one command
// buffer, and must give exactly the results of running them one by one —
// including across the operations that flush the batch (copyTensor records into
// it; readback / sync flush it).
const rand = (n: number, seed: number) => {
  const a = new Float32Array(n)
  let s = seed
  for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; a[i] = (s / 2 ** 32) * 2 - 1 }
  return a
}

function chain(b: WebGPUBackend) {
  const x = b.tensor(9, 7, 8, rand(9 * 7 * 8, 1))
  const w = { weights: rand(3 * 3 * 8 * 8, 2), bias: rand(8, 3) }
  const c1 = b.ops.Conv2d(x, w, { outChannels: 8, kernel: 3, stride: 1, padding: 'same', activation: 'relu' })
  const dw = b.ops.DepthwiseConv2d(c1.output, { weights: rand(9 * 8, 4), bias: rand(8, 5) }, { kernel: 3, stride: 2, padding: 'same', activation: 'hswish' })
  const pool = b.ops.GlobalAvgPool(dw.output)
  const sc = b.ops.ChannelScale(dw.output, pool.output)
  const add = b.ops.Add(sc.output, dw.output)
  return { x, ops: [c1, dw, pool, sc, add], out: add.output }
}

describe('WebGPU batch()', () => {
  it('batched run == op-by-op run', async () => {
    const b = await createWebGPUBackend()
    const g = chain(b)
    for (const op of g.ops) op.run()
    const ref = await b.readback(g.out)
    // Clobber the output, then rerun batched.
    const zero = b.tensor(g.out.h, g.out.w, g.out.c)
    b.copyTensor(zero, g.out)
    runBatched(b, () => { for (const op of g.ops) op.run() })
    const got = await b.readback(g.out)
    b.destroy()
    expect(Array.from(got)).toEqual(Array.from(ref))
  })

  it('copyTensor and readback inside a batch stay ordered', async () => {
    const b = await createWebGPUBackend()
    const g = chain(b)
    const held = b.tensor(g.out.h, g.out.w, g.out.c)
    let mid: Promise<Float32Array> | undefined
    runBatched(b, () => {
      for (const op of g.ops) op.run()
      b.copyTensor(g.out, held)          // recorded after the dispatches
      mid = b.readback(held)             // flushes: sees the copy
      const zero = b.tensor(g.out.h, g.out.w, g.out.c)
      b.copyTensor(zero, g.out)          // after the readback was issued
    })
    const ref = chain(b)
    for (const op of ref.ops) op.run()
    const want = await b.readback(ref.out)
    const got = await mid!
    const cleared = await b.readback(g.out)
    b.destroy()
    expect(Array.from(got)).toEqual(Array.from(want))
    expect(cleared.every(v => v === 0)).toBe(true)
  })

  it('nested batches submit once, at the outermost end', async () => {
    const b = await createWebGPUBackend()
    const g = chain(b)
    let submits = 0
    const q = b.device.queue
    const orig = q.submit.bind(q)
    q.submit = (bufs: Iterable<GPUCommandBuffer>) => { submits++; return orig(bufs) }
    runBatched(b, () => {
      runBatched(b, () => { g.ops[0].run(); g.ops[1].run() })
      expect(submits).toBe(0)
      for (const op of g.ops.slice(2)) op.run()
    })
    expect(submits).toBe(1)
    await b.sync()
    b.destroy()
  })
})
