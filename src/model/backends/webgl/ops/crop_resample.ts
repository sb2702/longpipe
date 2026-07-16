import type { Tensor, MLBuffer, CropResampleParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import shaderSrc from '~/model/backends/webgl/shaders/crop_resample.glsl'

// Box-driven square crop + bilinear resample + ((rgb - mean)/std) normalize.
export class CropResampleWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: MLBuffer[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = shaderSrc

  constructor(backend: WebGLBackend, frame: Tensor, box: Tensor, params: CropResampleParams) {
    super(backend)

    const slot = params.slot ?? 0
    if (slot >= box.w * box.h)
      throw new Error(`CropResample: slot ${slot} out of range for a ${box.h}×${box.w} box tensor`)

    const outTexture = this.makeTexture(null, params.outW, params.outH)
    this.output = { h: params.outH, w: params.outW, c: 4, texture: outTexture, texW: params.outW, texH: params.outH }
    this.inputs = [frame, box]

    this.samplers = [
      { name: 'u_frame', texture: (frame as WebGLTensor).texture },
      { name: 'u_box',   texture: (box as WebGLTensor).texture },
    ]
    this.uniformInts = { u_in_h: frame.h, u_in_w: frame.w, u_out_h: params.outH, u_out_w: params.outW, u_slot: slot }
    this.uniformFloats = {
      u_mean_r: params.mean[0], u_mean_g: params.mean[1], u_mean_b: params.mean[2],
      u_std_r:  params.std[0],  u_std_g:  params.std[1],  u_std_b:  params.std[2],
    }

    this.defaultSetup()
    this.dispatch = [params.outW, params.outH]
  }
}
