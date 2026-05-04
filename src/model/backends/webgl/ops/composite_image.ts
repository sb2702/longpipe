import type { Tensor } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'
import compositeImageSrc from '~/model/backends/webgl/shaders/composite_image.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Like CompositeSolidWebGL but the background is a Tensor (e.g. a virtual
// background image, or a blurred copy of the input).
export class CompositeImageWebGL {
  private readonly program: WebGLProgram
  private readonly imageTex: WebGLTexture
  private readonly alphaTex: WebGLTexture
  private readonly bgTex:    WebGLTexture

  constructor(
    private readonly backend: WebGLBackend,
    image: Tensor,
    alpha: Tensor,
    bg: Tensor,
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w
     || image.h !== bg.h    || image.w !== bg.w) {
      throw new Error(
        `CompositeImage: image (${image.h}×${image.w}), alpha (${alpha.h}×${alpha.w}), ` +
        `and bg (${bg.h}×${bg.w}) must all match. Run upscaler / resizer first.`,
      )
    }

    this.imageTex = (image as WebGLTensor).texture
    this.alphaTex = (alpha as WebGLTensor).texture
    this.bgTex    = (bg    as WebGLTensor).texture

    const gl = backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, compositeImageSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`composite_image GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`composite_image GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
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

    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, this.bgTex)
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_bg'), 2)

    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
