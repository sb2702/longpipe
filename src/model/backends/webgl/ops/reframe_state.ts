import type { Tensor, MLBuffer, ReframeStateParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/reframe_state.glsl'

// Auto-reframe camera state: (boxes, prev state, cmd) → new state (1×1×4).
export class ReframeStateWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number] = [1, 1]
  shader = shaderSrc

  constructor(backend: WebGLBackend, boxes: Tensor, prev: Tensor, cmd: Tensor, params: ReframeStateParams) {
    super(backend)
    const outTexture = this.makeTexture(null, 1, 1)
    this.output = { h: 1, w: 1, c: 4, texture: outTexture, texW: 1, texH: 1 }
    this.inputs = [boxes, prev, cmd]

    this.samplers = [
      { name: 'u_box',  texture: (boxes as WebGLTensor).texture },
      { name: 'u_prev', texture: (prev  as WebGLTensor).texture },
      { name: 'u_cmd',  texture: (cmd   as WebGLTensor).texture },
    ]
    this.uniformInts = { u_k: boxes.w * boxes.h }
    this.uniformFloats = {
      u_zoom: params.zoom, u_gravity: params.gravity, u_margin: params.margin,
      u_deadband: params.deadband, u_ease: params.ease, u_aspect: params.aspect,
    }
    this.defaultSetup()
  }
}
