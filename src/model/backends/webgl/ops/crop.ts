import type { Tensor, UpsampleParams, MLBuffer } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/crop.glsl'

export class CropWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, input: Tensor, params: UpsampleParams) {
    super(backend)
    const groups = input.c / 4
    const outTexW = params.outW * groups
    const outTexture = this.makeTexture(null, outTexW, params.outH)
    this.output = { h: params.outH, w: params.outW, c: input.c, texture: outTexture, texW: outTexW, texH: params.outH }
    this.inputs = [input]
    this.samplers = [{ name: 'u_input', texture: (input as WebGLTensor).texture }]
    this.defaultSetup()
    this.dispatch = [outTexW, params.outH]
  }
}
