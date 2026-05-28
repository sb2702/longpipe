import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import { toUploadView } from '~/utils/weights.ts'
import gammaSrc from '~/model/backends/webgl/shaders/gamma_residual.glsl'

// b_out = b + γ ⊙ h_new. γ is one f32 per channel, uploaded once at
// construction as a (c_groups, 1) RGBA texture.
export class GammaResidualWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = gammaSrc

  constructor(backend: WebGLBackend, b: Tensor, h_new: Tensor, gamma: ArrayLike<number>) {
    super(backend)

    const tb = b as WebGLTensor
    const channelGroups = b.c / 4

    const outTexture = this.makeTexture(null, tb.texW, tb.texH)
    this.output = { h: b.h, w: b.w, c: b.c, texture: outTexture, texW: tb.texW, texH: tb.texH }

    // γ as a (c_groups, 1) texture — one vec4 per channel group.
    // toUploadView coerces plain arrays (e.g. JSON-loaded number[]) into a
    // Float32Array view that gl.texImage2D accepts.
    const gammaTex = this.makeTexture(toUploadView(gamma), channelGroups, 1)

    this.inputs = [b, h_new]

    this.samplers = [
      { name: 'u_b',     texture: (b     as WebGLTensor).texture },
      { name: 'u_h_new', texture: (h_new as WebGLTensor).texture },
      { name: 'u_gamma', texture: gammaTex },
    ]

    this.uniformInts = {
      u_c_groups: channelGroups,
    }

    this.defaultSetup()
    this.dispatch = [tb.texW, tb.texH]
  }
}
