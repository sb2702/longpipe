import type { Tensor, MLBuffer, FaceBoxParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/face_box.glsl'

// Face-box decode: 5-keypoint heatmaps → 1×1×4 (cx, cy, halfSide, score) in
// frame fractions. One fragment decodes all 5 channels (grid is tiny).
export class FaceBoxWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number] = [1, 1]
  shader = shaderSrc

  constructor(backend: WebGLBackend, heatmaps: Tensor, params: FaceBoxParams) {
    super(backend)
    if (heatmaps.c !== 8)
      throw new Error(`FaceBoxFromHeatmaps: expected 8-ch heatmaps (5 real), got ${heatmaps.c}`)

    const outTexture = this.makeTexture(null, 1, 1)
    this.output = { h: 1, w: 1, c: 4, texture: outTexture, texW: 1, texH: 1 }
    this.inputs = [heatmaps]

    this.samplers = [{ name: 'u_hm', texture: (heatmaps as WebGLTensor).texture }]
    this.uniformInts = { u_h: heatmaps.h, u_w: heatmaps.w, u_win: params.win }
    this.uniformFloats = { u_thresh: params.thresh, u_box_scale: params.boxScale }

    this.defaultSetup()
  }
}
