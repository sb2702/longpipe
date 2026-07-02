import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'
import compositeTransparentSrc from '~/model/backends/webgl/shaders/composite_transparent.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Composites image + alpha over TRANSPARENCY (the matte becomes the canvas
// alpha channel), writes to the canvas (default framebuffer). Like
// CompositeSolidWebGL but with no background color — the subject is isolated so
// whatever sits behind the canvas shows through. Not a WebGLOp — produces no
// Tensor output, lives at the boundary between the model graph and the display
// surface.
//
// Caller invariants:
//   - image and alpha share h × w
//   - canvas.width === image.w, canvas.height === image.h
//     (the upscaler matches alpha to image res; this compositor does not
//     resample)
export class CompositeTransparentWebGL {
  private readonly program:  WebGLProgram
  private readonly imageTex: WebGLTexture
  private readonly alphaTex: WebGLTexture

  constructor(
    private readonly backend: WebGLBackend,
    image: Tensor,
    alpha: Tensor,
  ) {
    if (image.h !== alpha.h || image.w !== alpha.w)
      throw new Error(
        `CompositeTransparent: image (${image.h}×${image.w}) and alpha ` +
        `(${alpha.h}×${alpha.w}) must match. Run the upscaler first.`,
      )

    this.imageTex = (image as WebGLTensor).texture
    this.alphaTex = (alpha as WebGLTensor).texture

    const gl = backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, compositeTransparentSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`composite_transparent GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`composite_transparent GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
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

    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
