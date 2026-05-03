import { describe, it, expect } from 'vitest'
import type { Backend } from '~/model/backend'
import { WebGPUBackend } from '~/model/backends/webgpu/index'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { DecoderBlock } from '~/model/blocks/decoder_block'

import fixture from '../fixtures/decoder_block.json'

const THRESHOLD = 1e-4

const BACKENDS: Array<{ name: string; create: () => Promise<Backend> }> = [
  { name: 'WebGPU', create: () => WebGPUBackend.create() },
  { name: 'WebGL',  create: async () => WebGLBackend.create() },
]

describe.each(BACKENDS)('DecoderBlock ($name)', ({ create }) => {
  it('upsample+concat+2×conv matches PyTorch', async () => {
    const backend = await create()

    const [, deepC, deepH, deepW] = fixture.deep_shape
    const [, skipC, skipH, skipW] = fixture.skip_shape
    const deep = backend.tensor(deepH, deepW, deepC, new Float32Array(fixture.deep_input))
    const skip = backend.tensor(skipH, skipW, skipC, new Float32Array(fixture.skip_input))

    const block = new DecoderBlock(backend, deep, skip, { conv1: fixture.conv1, conv2: fixture.conv2 }, {
      outChannels: fixture.out_channels,
    })
    block.run()

    const result = await backend.readback(block.output)
    backend.destroy()

    const ref = new Float32Array(fixture.expected_output)
    let maxErr = 0
    for (let i = 0; i < ref.length; i++) maxErr = Math.max(maxErr, Math.abs(result[i] - ref[i]))
    expect(maxErr).toBeLessThan(THRESHOLD)
  })
})
