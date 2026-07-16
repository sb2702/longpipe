import type { Tensor, FaceTopology, FaceTouchupParams } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import type { WebGLTensor } from '~/model/backends/webgl/base_webgl_op.ts'

const ATLAS = 512

const QUAD_VERT = `#version 300 es
const vec2 VERTS[6] = vec2[6](
    vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0),
    vec2(-1.0,1.0),  vec2(1.0,-1.0), vec2(1.0,1.0)
);
void main() { gl_Position = vec4(VERTS[gl_VertexID], 0.0, 1.0); }`

// Landmark fetch + crop→frame transform, shared by the mesh vertex shaders.
// Landmarks: tensor texture (count/2, 1), two (x,y) pairs per texel. Box:
// 1×1 (cx, cy, halfSide/W, score) frame fractions.
const LM_COMMON = `
uniform sampler2D u_lm;
uniform sampler2D u_box;
uniform float u_thresh;
uniform float u_canvas_w;
uniform float u_canvas_h;

vec3 lmFrame(int i) {
    vec4 box = texelFetch(u_box, ivec2(0, 0), 0);
    vec4 g = texelFetch(u_lm, ivec2(i / 2, 0), 0);
    float lx = g[(i % 2) * 2];
    float ly = g[(i % 2) * 2 + 1];
    float hsx = box.z;
    float hsy = box.z * u_canvas_w / u_canvas_h;
    return vec3((box.x - hsx) + lx * 2.0 * hsx, (box.y - hsy) + ly * 2.0 * hsy, box.w);
}`

// Unwrap: vertex at canonical UV (atlas clip space), textured by the frame at
// the landmark's frame position. Atlas row = a_uv.y (no flip in FBO space).
const UNWRAP_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_uv;
layout(location = 1) in float a_idx;
${LM_COMMON}
out vec2 v_src;
void main() {
    vec3 l = lmFrame(int(a_idx));
    if (l.z < u_thresh) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    v_src = l.xy;
    gl_Position = vec4(a_uv * 2.0 - 1.0, 0.0, 1.0);
}`

// Manual bilinear over the frame TENSOR texture (NEAREST-filtered; texelFetch
// ignores filter state anyway) — same sampling as crop_resample.glsl.
const FRAME_BILINEAR = `
uniform sampler2D u_frame;
uniform float u_frame_w;
uniform float u_frame_h;

vec3 frameAt(int x, int y) {
    return texelFetch(u_frame, ivec2(clamp(x, 0, int(u_frame_w) - 1), clamp(y, 0, int(u_frame_h) - 1)), 0).rgb;
}
vec3 frameBilinear(vec2 f) {
    float sx = clamp(f.x * u_frame_w - 0.5, 0.0, u_frame_w - 1.0);
    float sy = clamp(f.y * u_frame_h - 0.5, 0.0, u_frame_h - 1.0);
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float tx = sx - float(x0);
    float ty = sy - float(y0);
    vec3 top = mix(frameAt(x0, y0),     frameAt(x0 + 1, y0),     tx);
    vec3 bot = mix(frameAt(x0, y0 + 1), frameAt(x0 + 1, y0 + 1), tx);
    return mix(top, bot, ty);
}`

const UNWRAP_FRAG = `#version 300 es
precision highp float;
precision highp int;
in vec2 v_src;
${FRAME_BILINEAR}
out vec4 fragColor;
void main() { fragColor = vec4(frameBilinear(v_src), 1.0); }`

// Separable gaussian on the atlas — uv from gl_FragCoord so read row == write
// row across passes (all atlas passes share texel space; no flips).
const BLUR_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_dir;
uniform float u_sigma;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / ${ATLAS}.0;
    float s = max(u_sigma, 0.001);
    int R = int(clamp(ceil(s * 2.5), 1.0, 48.0));
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -48; i <= 48; i++) {
        if (i < -R || i > R) continue;
        float w = exp(-float(i * i) / (2.0 * s * s));
        sum += texture(u_tex, uv + u_dir * float(i)) * w;
        wsum += w;
    }
    fragColor = sum / wsum;
}`

const COMBINE_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_atlas;
uniform sampler2D u_low;
uniform float u_detail;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / ${ATLAS}.0;
    vec4 a  = texture(u_atlas, uv);
    vec4 lo = texture(u_low, uv);
    fragColor = clamp(lo + (a - lo) * u_detail, 0.0, 1.0);
}`

// Single-pass edge-preserving bilateral (style 'bilateral'); range sigma 0.15.
const BILATERAL_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform float u_sigma;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / ${ATLAS}.0;
    vec2 texel = vec2(1.0 / ${ATLAS}.0);
    float ss = max(u_sigma, 0.001);
    int R = int(clamp(ceil(ss), 1.0, 12.0));
    float sr = 0.15;
    vec3 c = texture(u_tex, uv).rgb;
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int y = -12; y <= 12; y++) {
        if (y < -R || y > R) continue;
        for (int x = -12; x <= 12; x++) {
            if (x < -R || x > R) continue;
            vec3 sc = texture(u_tex, uv + texel * vec2(float(x), float(y))).rgb;
            float ws = exp(-float(x * x + y * y) / (2.0 * ss * ss));
            vec3 d = sc - c;
            float wr = exp(-dot(d, d) / (2.0 * sr * sr));
            sum += sc * ws * wr;
            wsum += ws * wr;
        }
    }
    fragColor = vec4(sum / wsum, 1.0);
}`

// Display blit (flips y — canvas is bottom-up, tensors top-down).
const BLIT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_image;
out vec4 fragColor;
void main() {
    int H = textureSize(u_image, 0).y;
    ivec2 px = ivec2(int(gl_FragCoord.x), H - 1 - int(gl_FragCoord.y));
    fragColor = vec4(texelFetch(u_image, px, 0).rgb, 1.0);
}`

// Composite mesh: positioned at landmark screen coords, sampling the smoothed
// atlas, blended by the weight mask × strength over the original pixels.
const COMP_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_uv;
layout(location = 1) in float a_idx;
${LM_COMMON}
out vec2 v_uv;
out vec2 v_src;
void main() {
    vec3 l = lmFrame(int(a_idx));
    if (l.z < u_thresh) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    v_uv = a_uv;
    v_src = l.xy;
    gl_Position = vec4(l.x * 2.0 - 1.0, 1.0 - 2.0 * l.y, 0.0, 1.0);
}`

const COMP_FRAG = `#version 300 es
precision highp float;
precision highp int;
in vec2 v_uv;
in vec2 v_src;
uniform sampler2D u_smoothed;
uniform sampler2D u_weight;
uniform float u_strength;
${FRAME_BILINEAR}
out vec4 fragColor;
void main() {
    vec3 orig = frameBilinear(v_src);
    vec3 sm   = texture(u_smoothed, v_uv).rgb;
    float w   = texture(u_weight, v_uv).r * u_strength;
    fragColor = vec4(mix(orig, sm, w), 1.0);
}`

function compileProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string, label: string): WebGLProgram {
  const mk = (type: number, src: string, what: string) => {
    const s = gl.createShader(type)!
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(`${label} ${what} compile error: ${gl.getShaderInfoLog(s)}`)
    return s
  }
  const prog = gl.createProgram()!
  gl.attachShader(prog, mk(gl.VERTEX_SHADER, vertSrc, 'vert'))
  gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, fragSrc, 'frag'))
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
    throw new Error(`${label} link error: ${gl.getProgramInfoLog(prog)}`)
  return prog
}

// UV-space face touch-up presenter (see face_touchup.wgsl for the pass math).
// All per-frame data (frame, landmarks, box) comes from SDK tensor textures —
// texelFetch in the vertex shaders, zero readback.
export class FaceTouchupWebGL {
  private readonly progs: Record<'unwrap' | 'blur' | 'combine' | 'bilateral' | 'blit' | 'comp', WebGLProgram>
  private readonly meshVao: WebGLVertexArrayObject
  private readonly fbo: WebGLFramebuffer
  private readonly atlas: WebGLTexture
  private readonly ping: WebGLTexture
  private readonly low: WebGLTexture
  private readonly smoothed: WebGLTexture
  private readonly weight: WebGLTexture
  private readonly frame: WebGLTensor
  private readonly lm: WebGLTensor
  private readonly box: WebGLTensor
  private readonly count: number
  private readonly params: FaceTouchupParams

  constructor(
    private readonly backend: WebGLBackend,
    frame: Tensor,
    landmarks: Tensor,
    box: Tensor,
    topo: FaceTopology,
    params: FaceTouchupParams,
  ) {
    const gl = backend.gl
    this.frame = frame as WebGLTensor
    this.lm = landmarks as WebGLTensor
    this.box = box as WebGLTensor
    this.count = topo.count
    this.params = params

    this.progs = {
      unwrap:    compileProgram(gl, UNWRAP_VERT, UNWRAP_FRAG, 'touchup unwrap'),
      blur:      compileProgram(gl, QUAD_VERT, BLUR_FRAG, 'touchup blur'),
      combine:   compileProgram(gl, QUAD_VERT, COMBINE_FRAG, 'touchup combine'),
      bilateral: compileProgram(gl, QUAD_VERT, BILATERAL_FRAG, 'touchup bilateral'),
      blit:      compileProgram(gl, QUAD_VERT, BLIT_FRAG, 'touchup blit'),
      comp:      compileProgram(gl, COMP_VERT, COMP_FRAG, 'touchup comp'),
    }

    // Static mesh VAO (uv → loc 0, idx → loc 1).
    this.meshVao = gl.createVertexArray()!
    gl.bindVertexArray(this.meshVao)
    const uvBuf = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf)
    gl.bufferData(gl.ARRAY_BUFFER, topo.uv, gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    const idxBuf = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf)
    gl.bufferData(gl.ARRAY_BUFFER, topo.idx, gl.STATIC_DRAW)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0)
    gl.bindVertexArray(null)

    // Atlas targets (RGBA8, LINEAR — sampled with texture()) + weight mask.
    const mkTex = (data: ImageBitmap | null) => {
      const t = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, t)
      if (data) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, data)
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, ATLAS, ATLAS, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      return t
    }
    this.atlas = mkTex(null)
    this.ping = mkTex(null)
    this.low = mkTex(null)
    this.smoothed = mkTex(null)
    this.weight = mkTex(topo.weightMask)
    this.fbo = gl.createFramebuffer()!
  }

  private bindTex(prog: WebGLProgram, unit: number, name: string, tex: WebGLTexture): void {
    const gl = this.backend.gl
    gl.activeTexture(gl.TEXTURE0 + unit)
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.uniform1i(gl.getUniformLocation(prog, name), unit)
  }

  private lmUniforms(prog: WebGLProgram): void {
    const gl = this.backend.gl
    this.bindTex(prog, 4, 'u_lm', this.lm.texture)
    this.bindTex(prog, 5, 'u_box', this.box.texture)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_thresh'), this.params.thresh)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_canvas_w'), this.frame.w)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_canvas_h'), this.frame.h)
  }

  private frameUniforms(prog: WebGLProgram): void {
    const gl = this.backend.gl
    this.bindTex(prog, 0, 'u_frame', this.frame.texture)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_frame_w'), this.frame.w)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_frame_h'), this.frame.h)
  }

  private atlasPass(target: WebGLTexture, draw: () => void): void {
    const gl = this.backend.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0)
    gl.viewport(0, 0, ATLAS, ATLAS)
    draw()
  }

  run(): void {
    const gl = this.backend.gl

    // 1. unwrap → atlas
    this.atlasPass(this.atlas, () => {
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.useProgram(this.progs.unwrap)
      this.frameUniforms(this.progs.unwrap)
      this.lmUniforms(this.progs.unwrap)
      gl.bindVertexArray(this.meshVao)
      gl.drawArrays(gl.TRIANGLES, 0, this.count)
      gl.bindVertexArray(null)
    })

    // 2–4. smoothing → `smoothed` (style-dependent).
    if ((this.params.style ?? 'freq-sep') === 'bilateral') {
      this.atlasPass(this.smoothed, () => {
        gl.useProgram(this.progs.bilateral)
        this.bindTex(this.progs.bilateral, 0, 'u_tex', this.atlas)
        gl.uniform1f(gl.getUniformLocation(this.progs.bilateral, 'u_sigma'), this.params.amount)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      })
    } else {
      const blur = (src: WebGLTexture, dst: WebGLTexture, dx: number, dy: number) =>
        this.atlasPass(dst, () => {
          gl.useProgram(this.progs.blur)
          this.bindTex(this.progs.blur, 0, 'u_tex', src)
          gl.uniform2f(gl.getUniformLocation(this.progs.blur, 'u_dir'), dx, dy)
          gl.uniform1f(gl.getUniformLocation(this.progs.blur, 'u_sigma'), this.params.amount)
          gl.drawArrays(gl.TRIANGLES, 0, 6)
        })
      blur(this.atlas, this.ping, 1 / ATLAS, 0)
      blur(this.ping, this.low, 0, 1 / ATLAS)

      this.atlasPass(this.smoothed, () => {
        gl.useProgram(this.progs.combine)
        this.bindTex(this.progs.combine, 0, 'u_atlas', this.atlas)
        this.bindTex(this.progs.combine, 1, 'u_low', this.low)
        gl.uniform1f(gl.getUniformLocation(this.progs.combine, 'u_detail'), this.params.detail)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      })
    }

    // 5. composite to canvas: frame blit, then the mesh over the face.
    this.backend.bindDisplayFramebuffer()
    gl.viewport(0, 0, this.frame.w, this.frame.h)
    gl.useProgram(this.progs.blit)
    this.bindTex(this.progs.blit, 0, 'u_image', this.frame.texture)
    gl.bindVertexArray(null)
    gl.drawArrays(gl.TRIANGLES, 0, 6)

    gl.useProgram(this.progs.comp)
    this.frameUniforms(this.progs.comp)
    this.lmUniforms(this.progs.comp)
    this.bindTex(this.progs.comp, 1, 'u_smoothed', this.smoothed)
    this.bindTex(this.progs.comp, 2, 'u_weight', this.weight)
    gl.uniform1f(gl.getUniformLocation(this.progs.comp, 'u_strength'), this.params.strength)
    gl.bindVertexArray(this.meshVao)
    gl.drawArrays(gl.TRIANGLES, 0, this.count)
    gl.bindVertexArray(null)
  }
}

// ── Tensor→Tensor stage form ─────────────────────────────────────────────────
// Renders the retouched frame INTO an output tensor's texture (WebGL tensors
// are textures, so this is direct — no unpack needed). Tensor-space targets
// don't y-flip: the display blit flips because the canvas is bottom-up; a
// tensor row 0 is just index 0, so blit copies rows straight through and the
// mesh positions map py → NDC py*2-1.
const BLIT_TENSOR_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_image;
out vec4 fragColor;
void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    fragColor = vec4(texelFetch(u_image, px, 0).rgb, 1.0);
}`

const COMP_TENSOR_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_uv;
layout(location = 1) in float a_idx;
${LM_COMMON}
out vec2 v_uv;
out vec2 v_src;
void main() {
    vec3 l = lmFrame(int(a_idx));
    if (l.z < u_thresh) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    v_uv = a_uv;
    v_src = l.xy;
    gl_Position = vec4(l.x * 2.0 - 1.0, l.y * 2.0 - 1.0, 0.0, 1.0);
}`

export class FaceTouchupStageWebGL {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  private readonly progs: Record<'unwrap' | 'blur' | 'combine' | 'bilateral' | 'blitT' | 'compT', WebGLProgram>
  private readonly meshVao: WebGLVertexArrayObject
  private readonly fbo: WebGLFramebuffer
  private readonly atlas: WebGLTexture
  private readonly ping: WebGLTexture
  private readonly low: WebGLTexture
  private readonly smoothed: WebGLTexture
  private readonly weightTex: WebGLTexture
  private readonly frame: WebGLTensor
  private readonly lm: WebGLTensor
  private readonly box: WebGLTensor
  private readonly count: number
  private readonly params: FaceTouchupParams

  constructor(
    private readonly backend: WebGLBackend,
    frame: Tensor,
    landmarks: Tensor,
    box: Tensor,
    topo: FaceTopology,
    params: FaceTouchupParams,
  ) {
    const gl = backend.gl
    this.inputs = [frame, landmarks, box]
    this.frame = frame as WebGLTensor
    this.lm = landmarks as WebGLTensor
    this.box = box as WebGLTensor
    this.count = topo.count
    this.params = params
    this.output = backend.tensor(frame.h, frame.w, 4) as WebGLTensor

    this.progs = {
      unwrap:    compileProgram(gl, UNWRAP_VERT, UNWRAP_FRAG, 'touchup-stage unwrap'),
      blur:      compileProgram(gl, QUAD_VERT, BLUR_FRAG, 'touchup-stage blur'),
      combine:   compileProgram(gl, QUAD_VERT, COMBINE_FRAG, 'touchup-stage combine'),
      bilateral: compileProgram(gl, QUAD_VERT, BILATERAL_FRAG, 'touchup-stage bilateral'),
      blitT:     compileProgram(gl, QUAD_VERT, BLIT_TENSOR_FRAG, 'touchup-stage blitT'),
      compT:     compileProgram(gl, COMP_TENSOR_VERT, COMP_FRAG, 'touchup-stage compT'),
    }

    this.meshVao = gl.createVertexArray()!
    gl.bindVertexArray(this.meshVao)
    const uvBuf = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf)
    gl.bufferData(gl.ARRAY_BUFFER, topo.uv, gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    const idxBuf = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf)
    gl.bufferData(gl.ARRAY_BUFFER, topo.idx, gl.STATIC_DRAW)
    gl.enableVertexAttribArray(1)
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0)
    gl.bindVertexArray(null)

    const mkTex = (data: ImageBitmap | null) => {
      const t = gl.createTexture()!
      gl.bindTexture(gl.TEXTURE_2D, t)
      if (data) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, data)
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, ATLAS, ATLAS, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      return t
    }
    this.atlas = mkTex(null)
    this.ping = mkTex(null)
    this.low = mkTex(null)
    this.smoothed = mkTex(null)
    this.weightTex = mkTex(topo.weightMask)
    this.fbo = gl.createFramebuffer()!
  }

  private bindTex(prog: WebGLProgram, unit: number, name: string, tex: WebGLTexture): void {
    const gl = this.backend.gl
    gl.activeTexture(gl.TEXTURE0 + unit)
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.uniform1i(gl.getUniformLocation(prog, name), unit)
  }

  private lmUniforms(prog: WebGLProgram): void {
    const gl = this.backend.gl
    this.bindTex(prog, 4, 'u_lm', this.lm.texture)
    this.bindTex(prog, 5, 'u_box', this.box.texture)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_thresh'), this.params.thresh)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_canvas_w'), this.frame.w)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_canvas_h'), this.frame.h)
  }

  private frameUniforms(prog: WebGLProgram): void {
    const gl = this.backend.gl
    this.bindTex(prog, 0, 'u_frame', this.frame.texture)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_frame_w'), this.frame.w)
    gl.uniform1f(gl.getUniformLocation(prog, 'u_frame_h'), this.frame.h)
  }

  private pass(target: WebGLTexture, w: number, h: number, draw: () => void): void {
    const gl = this.backend.gl
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0)
    gl.viewport(0, 0, w, h)
    draw()
  }

  run(): void {
    const gl = this.backend.gl

    this.pass(this.atlas, ATLAS, ATLAS, () => {
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.useProgram(this.progs.unwrap)
      this.frameUniforms(this.progs.unwrap)
      this.lmUniforms(this.progs.unwrap)
      gl.bindVertexArray(this.meshVao)
      gl.drawArrays(gl.TRIANGLES, 0, this.count)
      gl.bindVertexArray(null)
    })

    if ((this.params.style ?? 'freq-sep') === 'bilateral') {
      this.pass(this.smoothed, ATLAS, ATLAS, () => {
        gl.useProgram(this.progs.bilateral)
        this.bindTex(this.progs.bilateral, 0, 'u_tex', this.atlas)
        gl.uniform1f(gl.getUniformLocation(this.progs.bilateral, 'u_sigma'), this.params.amount)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      })
    } else {
      const blur = (src: WebGLTexture, dst: WebGLTexture, dx: number, dy: number) =>
        this.pass(dst, ATLAS, ATLAS, () => {
          gl.useProgram(this.progs.blur)
          this.bindTex(this.progs.blur, 0, 'u_tex', src)
          gl.uniform2f(gl.getUniformLocation(this.progs.blur, 'u_dir'), dx, dy)
          gl.uniform1f(gl.getUniformLocation(this.progs.blur, 'u_sigma'), this.params.amount)
          gl.drawArrays(gl.TRIANGLES, 0, 6)
        })
      blur(this.atlas, this.ping, 1 / ATLAS, 0)
      blur(this.ping, this.low, 0, 1 / ATLAS)
      this.pass(this.smoothed, ATLAS, ATLAS, () => {
        gl.useProgram(this.progs.combine)
        this.bindTex(this.progs.combine, 0, 'u_atlas', this.atlas)
        this.bindTex(this.progs.combine, 1, 'u_low', this.low)
        gl.uniform1f(gl.getUniformLocation(this.progs.combine, 'u_detail'), this.params.detail)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      })
    }

    // Composite into the output tensor: straight blit, then the mesh.
    this.pass(this.output.texture, this.frame.w, this.frame.h, () => {
      gl.useProgram(this.progs.blitT)
      this.bindTex(this.progs.blitT, 0, 'u_image', this.frame.texture)
      gl.bindVertexArray(null)
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      gl.useProgram(this.progs.compT)
      this.frameUniforms(this.progs.compT)
      this.lmUniforms(this.progs.compT)
      this.bindTex(this.progs.compT, 1, 'u_smoothed', this.smoothed)
      this.bindTex(this.progs.compT, 2, 'u_weight', this.weightTex)
      gl.uniform1f(gl.getUniformLocation(this.progs.compT, 'u_strength'), this.params.strength)
      gl.bindVertexArray(this.meshVao)
      gl.drawArrays(gl.TRIANGLES, 0, this.count)
      gl.bindVertexArray(null)
    })
  }
}
