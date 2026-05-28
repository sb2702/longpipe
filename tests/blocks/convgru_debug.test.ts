// Debug aid: runs the ConvGRU block on WebGL and reports max abs diff vs
// PyTorch reference at every intermediate. First failing step pinpoints the bug.
import { describe, it, expect } from 'vitest'
import { createWebGLBackend } from '../helpers/backends'
import { ConvGRU } from '~/model/blocks/convgru'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'

import fixture from '../fixtures/convgru_block.json'

const STEPS = [
  'cat_bh', 'z_pre', 'z', 'r_pre', 'r', 'rh',
  'cat_brh', 'cand_pre', 'h_til', 'h_new', 'b_out',
] as const

describe('ConvGRU debug (WebGL)', () => {
  it('intermediate-by-intermediate compare against PyTorch', async () => {
    const fx: any = fixture
    const backend = createWebGLBackend()

    const { passthrough, recurrent: c, height: H, width: W } = fx
    const a      = backend.tensor(H, W, passthrough, new Float32Array(fx.a))
    const b      = backend.tensor(H, W, c, new Float32Array(fx.b))
    const h_prev = backend.tensor(H, W, c, new Float32Array(fx.h_prev))

    const block = new ConvGRU(backend, a, b, h_prev, {
      zConv: fx.z_conv, rConv: fx.r_conv, cand: fx.cand, gamma: fx.gamma,
    }, { passthrough, recurrent: c })

    block.run()

    const diffs: Record<string, number> = {}
    for (const name of STEPS) {
      const gpu = await backend.readback(block.intermediates[name] as WebGLTensor)
      const ref = new Float32Array(fx.intermediates[name])
      let maxErr = 0
      for (let i = 0; i < ref.length; i++)
        maxErr = Math.max(maxErr, Math.abs(gpu[i] - ref[i]))
      diffs[name] = maxErr
    }

    // Print full table; first non-tiny diff is where the chain breaks.
    // (vitest captures console output and prints on test failure.)
    console.log('ConvGRU intermediates (WebGL vs PyTorch):')
    for (const name of STEPS) {
      console.log(`  ${name.padEnd(10)} maxErr = ${diffs[name].toExponential(3)}`)
    }

    backend.destroy()

    // Force the test to fail so the log shows up even when running headless.
    expect(diffs.b_out).toBeLessThan(1e-4)
  })
})
