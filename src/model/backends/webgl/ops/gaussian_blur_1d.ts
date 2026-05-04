import type { Tensor } from '~/model/backend'
import type { GaussianBlur1DParams } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op'
import blurSrc from '~/model/backends/webgl/shaders/gaussian_blur_1d.glsl'

export class GaussianBlur1DWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = blurSrc

  private readonly stepX:  number
  private readonly stepY:  number
  private readonly sigma:  number

  constructor(backend: WebGLBackend, input: Tensor, params: GaussianBlur1DParams) {
    super(backend)

    const ti      = input as WebGLTensor
    const cGroups = input.c / 4

    const outTexture = this.makeTexture(null, ti.texW, ti.texH)
    this.output = { h: input.h, w: input.w, c: input.c, texture: outTexture, texW: ti.texW, texH: ti.texH }
    this.inputs = [input]

    this.samplers = [{ name: 'u_input', texture: ti.texture }]
    this.uniformInts = {
      u_in_w:     input.w,
      u_in_h:     input.h,
      u_c_groups: cGroups,
    }

    this.stepX = params.direction === 'horizontal' ? 1 : 0
    this.stepY = params.direction === 'vertical'   ? 1 : 0
    this.sigma = params.sigma

    this.defaultSetup()
    this.dispatch = [ti.texW, ti.texH]
  }

  override run(): void {
    const gl = this.backend.gl
    gl.useProgram(this.program)
    // float and ivec2 uniforms aren't covered by the base class's uniformInts
    // helper — set them here before the base class's bindings + draw.
    gl.uniform2i(gl.getUniformLocation(this.program, 'u_step'), this.stepX, this.stepY)
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_sigma'), this.sigma)
    super.run()
  }
}
