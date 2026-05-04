import { describe, it, expect } from 'vitest'
import type { Backend, Tensor } from '~/model/backend'
import { EfficientNetLiteMattingSmall } from '~/model/networks/efficientnetlite_matting_small'
import type { ModelWeights } from '~/model/weights'

import fixture from '../fixtures/model_xs.json'

const THRESHOLD = 1e-3

import { BACKENDS } from '../helpers/backends'

// xs has the same architecture as small (small encoder, standard decoder trimmed),
// only the input shape differs (192×108 vs 256×144).
describe.each(BACKENDS)('EfficientNetLiteMattingSmall (xs preset, $name)', ({ create }) => {
  it('layer-by-layer outputs match PyTorch reference', async () => {
    const backend = await create()
    const { input_h, input_w } = fixture.config

    const input = backend.tensor(input_h, input_w, 4, new Float32Array(fixture.input))
    const model = new EfficientNetLiteMattingSmall(
      backend, input, fixture.weights as unknown as ModelWeights,
    )
    model.run()

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const m = model as any
    const caps = fixture.checkpoints as Record<string, number[]>

    const layers: Array<[string, { output: Tensor }, boolean?]> = [
      ['stem',       m.stem],
      ['s0',         m.s0],
      ['s1b0',       m.s1b0],
      ['s1b1',       m.s1b1],
      ['s2b0',       m.s2b0],
      ['s2b1',       m.s2b1],
      ['s3b0',       m.s3b0],
      ['s3b1',       m.s3b1],
      ['s3b2',       m.s3b2],
      ['s4b0',       m.s4b0],
      ['s4b1',       m.s4b1],
      ['s4b2',       m.s4b2],
      ['bottleneck', m.bottleneck],
      ['dec0',       m.dec0],
      ['dec1',       m.dec1],
      ['final_up',   m.finalUp],
      ['alpha',      m.alpha,   true],
    ]

    for (const [name, op, singleChannel] of layers) {
      if (!(name in caps)) continue

      const result = await backend.readback(op.output)
      const ref    = new Float32Array(caps[name])

      let maxErr = 0
      if (singleChannel) {
        const n = input_h * input_w
        for (let i = 0; i < n; i++)
          maxErr = Math.max(maxErr, Math.abs(result[i * 4] - ref[i]))
      } else {
        for (let i = 0; i < ref.length; i++)
          maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
      }

      expect(maxErr, `layer "${name}" max|err|`).toBeLessThan(THRESHOLD)
    }

    backend.destroy()
  })
})
