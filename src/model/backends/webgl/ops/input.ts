import type { ImageSource, InputOp } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'
import inputSrc from '~/model/backends/webgl/shaders/input.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Standalone (doesn't extend WebGLOp) because the source isn't a Tensor and
// the source texture needs LINEAR filtering rather than the NEAREST that
// WebGLOp.makeTexture defaults to. Builds its own program + source texture +
// output texture, but uses backend.tensor for the output so other ops can
// read it as a regular Tensor.
export class InputWebGL implements InputOp {
  readonly output: WebGLTensor

  private readonly gl: WebGL2RenderingContext
  private readonly program: WebGLProgram
  private readonly srcTex: WebGLTexture
  private readonly uOutW: WebGLUniformLocation
  private readonly uOutH: WebGLUniformLocation
  private readonly uSrc:  WebGLUniformLocation

  private source: ImageSource | null = null

  constructor(private readonly backend: WebGLBackend, h: number, w: number) {
    const gl = backend.gl
    this.gl = gl
    this.output = backend.tensor(h, w, 4)

    this.srcTex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S,     gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T,     gl.CLAMP_TO_EDGE)

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)
    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, inputSrc)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`Input GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`Input GLSL link error: ${gl.getProgramInfoLog(this.program)}`)

    this.uSrc  = gl.getUniformLocation(this.program, 'u_src')!
    this.uOutW = gl.getUniformLocation(this.program, 'u_out_w')!
    this.uOutH = gl.getUniformLocation(this.program, 'u_out_h')!
  }

  setSource(src: ImageSource): void {
    this.source = src
  }

  run(): void {
    if (!this.source) throw new Error('InputWebGL.run() called before setSource()')

    const gl = this.gl
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    // texImage2D accepts ImageBitmap and VideoFrame as TexImageSource.
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
      this.source as TexImageSource,
    )

    gl.useProgram(this.program)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex)
    gl.uniform1i(this.uSrc, 0)
    gl.uniform1i(this.uOutW, this.output.w)
    gl.uniform1i(this.uOutH, this.output.h)

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.backend.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.output.texture, 0)
    gl.viewport(0, 0, this.output.texW, this.output.texH)
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
