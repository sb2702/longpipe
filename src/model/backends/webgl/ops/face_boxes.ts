import type { Tensor, MLBuffer, FaceBoxesParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/face_boxes.glsl'

// Multi-face box decode: 5-keypoint heatmaps → 1×K×4 boxes, K = params.maxFaces.
// One fragment per face slot; each redoes the whole decode and emits its own
// slot (no shared memory in WebGL — see face_boxes.glsl).
export class FaceBoxesWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, heatmaps: Tensor, params: FaceBoxesParams) {
    super(backend)
    if (heatmaps.c !== 8)
      throw new Error(`FaceBoxesFromHeatmaps: expected 8-ch heatmaps (5 real), got ${heatmaps.c}`)
    if (params.maxFaces < 1 || params.maxFaces > 6)
      throw new Error(`FaceBoxesFromHeatmaps: maxFaces must be 1..6, got ${params.maxFaces}`)

    const K = params.maxFaces
    this.dispatch = [K, 1]
    const outTexture = this.makeTexture(null, K, 1)
    this.output = { h: 1, w: K, c: 4, texture: outTexture, texW: K, texH: 1 }
    this.inputs = [heatmaps]

    this.samplers = [{ name: 'u_hm', texture: (heatmaps as WebGLTensor).texture }]
    this.uniformInts = { u_h: heatmaps.h, u_w: heatmaps.w, u_win: params.win, u_max_faces: K }
    this.uniformFloats = { u_thresh: params.thresh, u_box_scale: params.boxScale, u_tol: params.tol }

    this.defaultSetup()
  }
}
