import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { ConvGRU } from '~/model/blocks/convgru'
import { BACKENDS } from '../helpers/backends'

import fixture from '../fixtures/gru_fused.json'

const THRESHOLD = 1e-4

// Production ConvGRU (c_up=2, recurrent=1): GatesFused → CandUpdateFused chained.
// uIn carries (a, b) in .xy; hPrev carries the hidden in .z; output is
// (a, b_out, h_new, 0).
async function runFixture(backend: Backend) {
  const { height: H, width: W } = fixture
  const uIn   = backend.tensor(H, W, 4, new Float32Array(fixture.u_in))
  const hPrev = backend.tensor(H, W, 4, new Float32Array(fixture.h_prev))

  const block = new ConvGRU(backend, uIn, hPrev, {
    gates:     fixture.gates_weights,
    gatesBias: fixture.gates_bias,
    cand:      fixture.cand_weights,
    candBias:  fixture.cand_bias,
    gamma:     fixture.gamma,
  })
  block.run()

  const result = await backend.readback(block.output)
  const ref = new Float32Array(fixture.expected_output)
  let maxErr = 0
  for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
  return maxErr
}

describe.each(BACKENDS)('ConvGRU ($name)', ({ create }) => {
  it('gates + cand_update chained match PyTorch (incl. h_new in .z)', async () => {
    const backend = await create()
    expect(await runFixture(backend)).toBeLessThan(THRESHOLD)
    backend.destroy()
  })
})
