import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import gruUpdateSrc from '~/model/backends/webgl/shaders/gru_update.glsl'

export class GruUpdateWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = gruUpdateSrc

  constructor(backend: WebGLBackend, z: Tensor, h_prev: Tensor, h_til: Tensor) {
    super(backend)

    const tz = z as WebGLTensor

    const outTexture = this.makeTexture(null, tz.texW, tz.texH)
    this.output = { h: z.h, w: z.w, c: z.c, texture: outTexture, texW: tz.texW, texH: tz.texH }

    this.inputs = [z, h_prev, h_til]

    this.samplers = [
      { name: 'u_z',      texture: (z      as WebGLTensor).texture },
      { name: 'u_h_prev', texture: (h_prev as WebGLTensor).texture },
      { name: 'u_h_til',  texture: (h_til  as WebGLTensor).texture },
    ]

    this.defaultSetup()
    this.dispatch = [tz.texW, tz.texH]
  }
}
