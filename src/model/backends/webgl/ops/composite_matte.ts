import type { Tensor } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'
import compositeMatteSrc from '~/model/backends/webgl/shaders/composite_matte.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Renders the raw 1-channel alpha matte as a premultiplied white silhouette to
// the canvas (default framebuffer). Alpha only — no image, no background. Not a
// WebGLOp — produces no Tensor output, lives at the boundary between the model
// graph and the display surface.
//
// Caller invariants:
//   - canvas.width === alpha.w, canvas.height === alpha.h (no resampling here)
export class CompositeMatteWebGL {
  private readonly program:  WebGLProgram
  private readonly alphaTex: WebGLTexture

  constructor(
    private readonly backend: WebGLBackend,
    alpha: Tensor,
  ) {
    this.alphaTex = (alpha as WebGLTensor).texture

    const gl = backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, compositeMatteSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`composite_matte GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`composite_matte GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
  }

  run(): void {
    const gl = this.backend.gl
    gl.useProgram(this.program)

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.alphaTex)
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_alpha'), 0)

    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
