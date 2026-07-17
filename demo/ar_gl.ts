// WebGL2 scene for the face-AR PoC: video quad → depth-only face occluder →
// glasses. Deliberately a SEPARATE context from the SDK backend — we read the
// landmarks back to the CPU anyway, and the video is uploaded here directly, so
// nothing needs sharing and nothing conflicts.
//
// The camera is a TRUE PERSPECTIVE pose from POSIT: rotation R, translation t,
// focal f. Clip coords are computed straight from the camera-space point rather
// than via a projection matrix — that dodges every GL handedness/convention trap,
// and setting w = Zc keeps interpolation perspective-correct.
//
// (This replaced a scaled-orthographic fit, which was measurably wrong: 6° of
// pose error head-on and 14° at 45° yaw with PERFECT landmarks. POSIT is 0.00°.)
//
// Occlusion is the thing that separates "glasses on a face" from "glasses pasted
// on a photo" (it's what jeeliz's occluderURL does): the canonical face mesh is
// drawn with colorMask off so it writes depth only, and the far temple then fails
// the depth test where it would otherwise track across the cheek.

export interface Pose {
  r0: [number, number, number]   // rotation rows, model → camera
  r1: [number, number, number]
  r2: [number, number, number]   // the optical axis: Zc grows with distance
  t:  [number, number, number]   // camera-space position of the model centre
  center: [number, number, number]   // model centroid, subtracted before rotating
  f:  number                     // focal length, canvas px
  pp: [number, number]           // principal point, canvas px
}

const QUAD_VS = `#version 300 es
out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`

const QUAD_FS = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 o;
void main() { o = vec4(texture(u_tex, v_uv).rgb, 1.0); }`

// Shared by the occluder and the glasses. Rotation is passed as three ROW vec3s,
// not a mat3 — GLSL's mat3 is column-major and that trap isn't worth the risk.
const MESH_VS = `#version 300 es
precision highp float;
in vec3 a_pos;
in vec3 a_normal;
uniform vec3 u_r0, u_r1, u_r2, u_t, u_center;
uniform vec2 u_canvas, u_pp;
uniform float u_f, u_near, u_far;
out float v_shade;
void main() {
  vec3 q  = a_pos - u_center;
  vec3 Xc = vec3(dot(u_r0, q), dot(u_r1, q), dot(u_r2, q)) + u_t;
  float zc = max(Xc.z, 1e-4);

  // Pinhole, in canvas px. y flips here — image y is down; the FIT works in a
  // standard camera frame and never sees this.
  float sx = u_pp.x + u_f * Xc.x / zc;
  float sy = u_pp.y - u_f * Xc.y / zc;
  float ndcx = sx / u_canvas.x * 2.0 - 1.0;
  float ndcy = 1.0 - sy / u_canvas.y * 2.0;
  float ndcz = ((zc - u_near) / (u_far - u_near)) * 2.0 - 1.0;
  // w = zc so the rasterizer interpolates perspective-correctly.
  gl_Position = vec4(vec3(ndcx, ndcy, ndcz) * zc, zc);

  vec3 nc = vec3(dot(u_r0, a_normal), dot(u_r1, a_normal), dot(u_r2, a_normal));
  // abs(), not -nc.z: jeeliz's meshes carry no normals and their winding is
  // unverified, so shade two-sided rather than have half the model go black.
  v_shade = clamp(abs(normalize(nc).z), 0.0, 1.0) * 0.72 + 0.28;
}`

const MESH_FS = `#version 300 es
precision highp float;
uniform vec4 u_color;
in float v_shade;
out vec4 o;
void main() { o = vec4(u_color.rgb * v_shade, u_color.a); }`

function compile(gl: WebGL2RenderingContext, vs: string, fs: string, label: string): WebGLProgram {
  const mk = (type: number, src: string) => {
    const s = gl.createShader(type)!
    gl.shaderSource(s, src); gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(`${label}: ${gl.getShaderInfoLog(s)}`)
    return s
  }
  const p = gl.createProgram()!
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs))
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs))
  gl.linkProgram(p)
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`${label}: ${gl.getProgramInfoLog(p)}`)
  return p
}

export interface Mesh { vao: WebGLVertexArrayObject; count: number }

export class ArScene {
  private readonly gl: WebGL2RenderingContext
  private readonly quadProg: WebGLProgram
  private readonly meshProg: WebGLProgram
  private readonly videoTex: WebGLTexture

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false })
    if (!gl) throw new Error('WebGL2 unavailable for the AR scene')
    this.gl = gl
    this.quadProg = compile(gl, QUAD_VS, QUAD_FS, 'ar quad')
    this.meshProg = compile(gl, MESH_VS, MESH_FS, 'ar mesh')
    this.videoTex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR)
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE)
  }

  // Non-indexed mesh: flat arrays of positions + normals.
  makeMesh(pos: Float32Array, normal: Float32Array): Mesh {
    const gl = this.gl
    const vao = gl.createVertexArray()!
    gl.bindVertexArray(vao)
    for (const [loc, data] of [[0, pos], [1, normal]] as Array<[number, Float32Array]>) {
      const b = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, b)
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW)
      gl.enableVertexAttribArray(loc)
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0)
    }
    gl.bindVertexArray(null)
    return { vao, count: pos.length / 3 }
  }

  private setPose(pose: Pose): void {
    const gl = this.gl, p = this.meshProg
    const u = (n: string) => gl.getUniformLocation(p, n)
    gl.useProgram(p)
    gl.uniform3fv(u('u_r0'), pose.r0); gl.uniform3fv(u('u_r1'), pose.r1); gl.uniform3fv(u('u_r2'), pose.r2)
    gl.uniform3fv(u('u_t'), pose.t)
    gl.uniform3fv(u('u_center'), pose.center)
    gl.uniform2f(u('u_canvas'), gl.canvas.width, gl.canvas.height)
    gl.uniform2fv(u('u_pp'), pose.pp)
    gl.uniform1f(u('u_f'), pose.f)
    // Depth range bracketing the head at its fitted distance.
    gl.uniform1f(u('u_near'), Math.max(1e-3, pose.t[2] * 0.3))
    gl.uniform1f(u('u_far'), pose.t[2] * 2.0)
  }

  private drawMesh(m: Mesh, color: [number, number, number, number]): void {
    const gl = this.gl
    gl.uniform4fv(gl.getUniformLocation(this.meshProg, 'u_color'), color)
    gl.bindVertexArray(m.vao)
    gl.drawArrays(gl.TRIANGLES, 0, m.count)
    gl.bindVertexArray(null)
  }

  // One frame. `parts` are opaque (frame, temples); `lenses` blend last.
  render(
    frame: ImageBitmap,
    pose: Pose | null,
    occluder: Mesh | null,
    parts: Array<{ mesh: Mesh; color: [number, number, number, number] }>,
    lenses: Array<{ mesh: Mesh; color: [number, number, number, number] }>,
  ): void {
    const gl = this.gl
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.disable(gl.DEPTH_TEST)
    gl.depthMask(false)
    gl.clearColor(0, 0, 0, 1)
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

    // Video backdrop.
    gl.useProgram(this.quadProg)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame)
    gl.uniform1i(gl.getUniformLocation(this.quadProg, 'u_tex'), 0)
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 3)

    if (!pose) return

    gl.enable(gl.DEPTH_TEST)
    gl.depthFunc(gl.LESS)
    gl.depthMask(true)
    this.setPose(pose)

    // Depth-only face: writes z, paints nothing. This is what hides the far
    // temple behind the head.
    if (occluder) {
      gl.colorMask(false, false, false, false)
      this.drawMesh(occluder, [0, 0, 0, 1])
      gl.colorMask(true, true, true, true)
    }

    for (const { mesh, color } of parts) this.drawMesh(mesh, color)

    // Lenses blend, so they must not write depth (order-dependent otherwise).
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.depthMask(false)
    for (const { mesh, color } of lenses) this.drawMesh(mesh, color)
    gl.disable(gl.BLEND)
    gl.depthMask(true)
  }
}
