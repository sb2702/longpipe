import type { Tensor, LandmarkOverlayParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'
import passthroughSrc from '~/model/backends/webgl/shaders/composite_passthrough.glsl'

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Vertex-pulled landmark points: gl_VertexID indexes the landmark tensor
// (texelFetch in the VERTEX shader — the per-frame data never leaves the GPU).
// Landmark tensor is 1×1×(count·2) → texture (count/2, 1), two (x,y) pairs per
// texel. Box tensor is 1×1: (cx, cy, halfSide/W, score) frame fractions.
// The blit pass flips y (canvas bottom-up vs tensor top-down), so image-frac
// py maps to NDC 1-2·py — same formula as the WebGPU overlay.
const POINTS_VERT = `#version 300 es
precision highp float;
uniform sampler2D u_lm;
uniform sampler2D u_box;
uniform int u_slot;
uniform float u_thresh;
uniform float u_point_size;
uniform float u_canvas_w;
uniform float u_canvas_h;

void main() {
    int i = gl_VertexID;
    vec4 box = texelFetch(u_box, ivec2(u_slot, 0), 0);
    if (box.w < u_thresh) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // clipped away
        gl_PointSize = 0.0;
        return;
    }
    vec4 g = texelFetch(u_lm, ivec2(i / 2, 0), 0);
    float lx = g[(i % 2) * 2];
    float ly = g[(i % 2) * 2 + 1];

    float hsx = box.z;
    float hsy = box.z * u_canvas_w / u_canvas_h;
    float px = (box.x - hsx) + lx * 2.0 * hsx;
    float py = (box.y - hsy) + ly * 2.0 * hsy;

    gl_Position = vec4(px * 2.0 - 1.0, 1.0 - 2.0 * py, 0.0, 1.0);
    gl_PointSize = u_point_size;
}`

const POINTS_FRAG = `#version 300 es
precision highp float;
uniform vec3 u_color;
out vec4 fragColor;
void main() { fragColor = vec4(u_color, 1.0); }`

function compileProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string, label: string): WebGLProgram {
  const vert = gl.createShader(gl.VERTEX_SHADER)!
  gl.shaderSource(vert, vertSrc)
  gl.compileShader(vert)
  if (!gl.getShaderParameter(vert, gl.COMPILE_STATUS))
    throw new Error(`${label} vert compile error: ${gl.getShaderInfoLog(vert)}`)
  const frag = gl.createShader(gl.FRAGMENT_SHADER)!
  gl.shaderSource(frag, fragSrc)
  gl.compileShader(frag)
  if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS))
    throw new Error(`${label} frag compile error: ${gl.getShaderInfoLog(frag)}`)
  const prog = gl.createProgram()!
  gl.attachShader(prog, vert)
  gl.attachShader(prog, frag)
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
    throw new Error(`${label} link error: ${gl.getProgramInfoLog(prog)}`)
  return prog
}

// Landmark overlay presenter — image blit + vertex-pulled landmark points onto
// the canvas (default framebuffer). No readback anywhere in the path.
export class LandmarkOverlayWebGL {
  private readonly imgProgram: WebGLProgram
  private readonly ptsProgram: WebGLProgram
  private readonly imageTex: WebGLTexture
  private readonly lmTex: WebGLTexture
  private readonly boxTex: WebGLTexture
  private readonly params: LandmarkOverlayParams
  private readonly canvasW: number
  private readonly canvasH: number
  private readonly slot: number

  constructor(
    private readonly backend: WebGLBackend,
    image: Tensor,
    landmarks: Tensor,
    box: Tensor,
    params: LandmarkOverlayParams,
  ) {
    if (landmarks.c < params.count * 2)
      throw new Error(`LandmarkOverlay: landmarks tensor holds ${landmarks.c / 2} points < count ${params.count}`)
    this.slot = params.slot ?? 0
    if (this.slot >= box.w * box.h)
      throw new Error(`LandmarkOverlay: slot ${this.slot} out of range for a ${box.h}×${box.w} box tensor`)
    this.imageTex = (image as WebGLTensor).texture
    this.lmTex = (landmarks as WebGLTensor).texture
    this.boxTex = (box as WebGLTensor).texture
    this.params = params
    this.canvasW = image.w
    this.canvasH = image.h

    const gl = backend.gl
    this.imgProgram = compileProgram(gl, QUAD_VERT, passthroughSrc, 'landmark_overlay img')
    this.ptsProgram = compileProgram(gl, POINTS_VERT, POINTS_FRAG, 'landmark_overlay pts')
  }

  run(): void {
    const gl = this.backend.gl
    this.backend.bindDisplayFramebuffer()
    gl.bindVertexArray(null)

    // drawImage=false layers this face's dots onto what a previous overlay
    // already blitted (multi-face) — the image blit is what would wipe them.
    if (this.params.drawImage ?? true) {
      gl.useProgram(this.imgProgram)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, this.imageTex)
      gl.uniform1i(gl.getUniformLocation(this.imgProgram, 'u_image'), 0)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
    }

    gl.useProgram(this.ptsProgram)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.lmTex)
    gl.uniform1i(gl.getUniformLocation(this.ptsProgram, 'u_lm'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.boxTex)
    gl.uniform1i(gl.getUniformLocation(this.ptsProgram, 'u_box'), 1)
    gl.uniform1i(gl.getUniformLocation(this.ptsProgram, 'u_slot'), this.slot)
    gl.uniform1f(gl.getUniformLocation(this.ptsProgram, 'u_thresh'), this.params.thresh)
    gl.uniform1f(gl.getUniformLocation(this.ptsProgram, 'u_point_size'), this.params.pointSize)
    gl.uniform1f(gl.getUniformLocation(this.ptsProgram, 'u_canvas_w'), this.canvasW)
    gl.uniform1f(gl.getUniformLocation(this.ptsProgram, 'u_canvas_h'), this.canvasH)
    gl.uniform3f(gl.getUniformLocation(this.ptsProgram, 'u_color'),
      this.params.color[0], this.params.color[1], this.params.color[2])
    gl.drawArrays(gl.POINTS, 0, this.params.count)
  }
}
