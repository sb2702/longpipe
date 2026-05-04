import type { Tensor } from '~/model/backend'
import type { WebGLBackend } from '~/model/backends/webgl/index'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op'
import compositePassthroughSrc from '~/model/backends/webgl/shaders/composite_passthrough.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Passthrough "compositor": writes image directly to the canvas (default
// framebuffer). No alpha, no background. Not a WebGLOp — produces no
// Tensor output, lives at the boundary between the model graph and the
// display surface.
//
// Used by RenderOp when the renderer is in disabled state.
//
// Caller invariants:
//   - canvas.width === image.w, canvas.height === image.h
export class CompositePassthroughWebGL {
  private readonly program:  WebGLProgram
  private readonly imageTex: WebGLTexture

  constructor(
    private readonly backend: WebGLBackend,
    image: Tensor,
  ) {
    this.imageTex = (image as WebGLTensor).texture

    const gl = backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, compositePassthroughSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`composite_passthrough GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`composite_passthrough GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
  }

  run(): void {
    const gl = this.backend.gl
    gl.useProgram(this.program)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.imageTex)
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_image'), 0)

    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
