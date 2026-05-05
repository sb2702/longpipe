import type { Tensor, MLBuffer, Op, DataView_ } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'

export interface WebGLTensor extends Tensor {
  readonly texture: WebGLTexture
  readonly texW: number  // w * (c/4)
  readonly texH: number  // h
}

// Stores raw data; ops create textures with backend-specific dimensions.
// `data` may be Float32 (fp32 source) or Uint16 (raw fp16 bits from a .f16.bin
// loader) — the backend's textureFormat decides how it's uploaded.
export interface WebGLMLBuffer extends MLBuffer {
  readonly data: DataView_
}

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Base for render-to-texture ops. Binding order: inputs → weights → (uniforms via gl.uniform1i).
export abstract class WebGLOp implements Op {
  abstract readonly inputs: Tensor[]
  abstract readonly weights: MLBuffer[]
  abstract readonly output: WebGLTensor
  protected abstract dispatch: [number, number]  // [texW, texH] for viewport

  shader: string = ''
  protected program!: WebGLProgram
  protected samplers: Array<{ name: string; texture: WebGLTexture }> = []
  protected uniformInts: Record<string, number> = {}

  constructor(protected readonly backend: WebGLBackend) {}

  // Create a texture in the backend's current format. Source data may be fp32
  // or fp16-bits or null (allocate-only). Conversion to the backend dtype is
  // delegated to backend.toTextureView().
  protected makeTexture(data: DataView_ | null, w: number, h: number): WebGLTexture {
    const gl  = this.backend.gl
    const fmt = this.backend.textureFormat
    const view = this.backend.toTextureView(data)
    const tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internalFormat, w, h, 0, fmt.format, fmt.type, view)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    return tex
  }

  protected defaultSetup(): void {
    const gl = this.backend.gl

    const vert = gl.createShader(gl.VERTEX_SHADER)!
    gl.shaderSource(vert, QUAD_VERT)
    gl.compileShader(vert)

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!
    gl.shaderSource(frag, this.shader)
    gl.compileShader(frag)
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
      throw new Error(`GLSL compile error: ${gl.getShaderInfoLog(frag)}`)

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vert)
    gl.attachShader(this.program, frag)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(`GLSL link error: ${gl.getProgramInfoLog(this.program)}`)
  }

  run(): void {
    const gl = this.backend.gl
    gl.useProgram(this.program)

    // Bind textures in order
    this.samplers.forEach(({ name, texture }, i) => {
      gl.activeTexture(gl.TEXTURE0 + i)
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.uniform1i(gl.getUniformLocation(this.program, name), i)
    })

    // Set integer uniforms
    for (const [name, val] of Object.entries(this.uniformInts)) {
      gl.uniform1i(gl.getUniformLocation(this.program, name), val)
    }

    // Render to output texture
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.backend.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.output.texture, 0)
    gl.viewport(0, 0, this.dispatch[0], this.dispatch[1])
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
  }
}
