import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import channelScaleSrc from '~/model/backends/webgl/shaders/channel_scale.glsl'

// x (H, W, C) * scale (1, 1, C), broadcast over H×W.
export class ChannelScaleWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = channelScaleSrc

  constructor(backend: WebGLBackend, x: Tensor, scale: Tensor) {
    super(backend)
    if (scale.h !== 1 || scale.w !== 1 || scale.c !== x.c)
      throw new Error(`ChannelScale: scale must be 1×1×${x.c}, got ${scale.h}×${scale.w}×${scale.c}`)
    const tx = x as WebGLTensor
    this.output = { h: x.h, w: x.w, c: x.c, texture: this.makeTexture(null, tx.texW, tx.texH), texW: tx.texW, texH: tx.texH }
    this.inputs = [x, scale]
    this.samplers = [
      { name: 'u_input', texture: tx.texture },
      { name: 'u_scale', texture: (scale as WebGLTensor).texture },
    ]
    this.uniformInts = { u_c_groups: x.c / 4 }
    this.defaultSetup()
    this.dispatch = [tx.texW, tx.texH]
  }
}
