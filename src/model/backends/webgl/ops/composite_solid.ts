import type { Tensor } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'
import compositeSolidSrc from '~/model/backends/webgl/shaders/composite_solid.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Composites image + alpha over a solid background, writes to the canvas
// (default framebuffer). Not a WebGLOp — produces no Tensor output, lives at
// the boundary between the model graph and the display surface.
//
// Caller invariants:
//   - image and alpha share h × w
//   - canvas.width === image.w, canvas.height === image.h
//     (the upscaler is responsible for matching alpha to image res; this
//     compositor does not resample)
export class CompositeSolidWebGL {
  private readonly program: WebGLProgram
  private readonly imageTex: WebGLTexture
  private readonly alphaTex: WebGLTexture
  private readonly bgColor:  [number, number, number]

  constructor(
    private readonly backend: WebGLBackend,
    image: Tensor,
    alpha: Tensor,
    bgColor: [number, number, number],
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w)
      throw new Error(
        `CompositeSolid: image (${image.h}×${image.w}) and alpha ` +
        `(${alpha.h}×${alpha.w}) must match. Run the upscaler first.`,
      )

    const ti = image as WebGLTensor
    const ta = alpha as WebGLTensor
    this.imageTex = ti.texture
    this.alphaTex = ta.texture
    this.bgColor  = bgColor

    const gl = backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, compositeSolidSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`composite_solid GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`composite_solid GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
  }

  run(): void {
    const gl = this.backend.gl
    gl.useProgram(this.program)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.imageTex)
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_image'), 0)

    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.alphaTex)
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_alpha'), 1)

    gl.uniform3f(
      gl.getUniformLocation(this.program, 'u_bgColor'),
      this.bgColor[0], this.bgColor[1], this.bgColor[2],
    )

    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
