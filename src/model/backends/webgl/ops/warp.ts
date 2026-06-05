import type { Tensor, MLBuffer, WarpParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/warp.glsl'

// Bilinear gather-warp. Source + flow are 4-ch (1 group) — texW = W.
export class WarpWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, source: Tensor, flow: Tensor, params: WarpParams) {
    super(backend)

    const W = source.w, H = source.h
    const outTexture = this.makeTexture(null, W, H)
    this.output = { h: H, w: W, c: source.c, texture: outTexture, texW: W, texH: H }
    this.inputs = [source, flow]

    this.samplers = [
      { name: 'u_source', texture: (source as WebGLTensor).texture },
      { name: 'u_flow',   texture: (flow as WebGLTensor).texture },
    ]
    this.uniformInts   = { u_w: W, u_h: H }
    this.uniformFloats = { u_flow_scale: params.flowScale }

    this.defaultSetup()
    this.dispatch = [W, H]
  }
}
