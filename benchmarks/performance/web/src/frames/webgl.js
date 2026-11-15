// Adapted from github.com/sb2702/webrtc-hacks-webgpu (segmentation/web/src/frames/webgl.js,
// commit 2c6528c, MIT, (c) 2026 Sam Bhattacharyya). Change: the small GL helper the
// original borrowed from that repo's WebGL network backend (program cache, drawTo,
// resetState) is inlined below as GLHelper, since there is no network here.
//
// WebGL frame I/O: VideoFrame in -> model input tensor, mask -> VideoFrame out.
//
// Shares a WebGLBackend's context (and its program cache). Used by our WebGL
// network and by MediaPipe's GPU delegate, which hands back its mask as a
// WebGLTexture in the same context.

import { frameSize } from './webgpu.js'

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 result;
`
const VERTEX = `#version 300 es
void main() {
  vec2 xy = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(xy * 2.0 - 1.0, 0.0, 1.0);
}`

export class GLHelper {
  constructor(gl) {
    this.gl = gl
    this.programs = new Map()
    // Shared libraries (MediaPipe) may leave a VAO bound with attributes
    // enabled; our full-screen triangles use no attributes at all.
    this.vao = gl.createVertexArray()
  }

  program(name, _act, source) {
    if (!this.programs.has(name)) {
      const gl = this.gl
      const compile = (type, src) => {
        const shader = gl.createShader(type)
        gl.shaderSource(shader, src)
        gl.compileShader(shader)
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(`${name}: ${gl.getShaderInfoLog(shader)}`)
        return shader
      }
      const program = gl.createProgram()
      gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX))
      gl.attachShader(program, compile(gl.FRAGMENT_SHADER, HEADER + source))
      gl.linkProgram(program)
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`${name}: ${gl.getProgramInfoLog(program)}`)
      this.programs.set(name, program)
    }
    return this.programs.get(name)
  }

  drawTo(target, w, h) {
    const gl = this.gl
    gl.bindVertexArray(this.vao)
    gl.bindFramebuffer(gl.FRAMEBUFFER, target)
    gl.viewport(0, 0, w, h)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  // Undo any fixed-function state another user of the context left enabled.
  resetState() {
    const gl = this.gl
    for (const cap of [gl.BLEND, gl.SCISSOR_TEST, gl.DEPTH_TEST, gl.STENCIL_TEST, gl.CULL_FACE]) gl.disable(cap)
  }
}

const SHADERS = {
  // Same 4-tap box downscale as the WebGPU path. Writes vec4 tensor texels
  // (RGB + a zero pad channel).
  preprocess: `
uniform sampler2D frame;
uniform vec2 size;
void main() {
  vec2 texel = 1.0 / size;
  vec2 uv = gl_FragCoord.xy * texel;
  vec3 rgb = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    vec2 offset = (vec2(i & 1, i >> 1) - 0.5) * 0.5 * texel;
    rgb += texture(frame, uv + offset).rgb;
  }
  result = vec4(rgb * 0.25, 0.0);
}`,

  // mix(color, frame, mask), drawn to the canvas. Float textures aren't
  // guaranteed to be filterable, so the mask is filtered by hand. `flip_mask`
  // handles masks stored bottom-up (GL convention) rather than top-down.
  composite: `
uniform sampler2D frame, mask;
uniform vec2 out_size;
uniform vec3 color;
uniform int mask_w, mask_h, flip_mask;
float mask_at(int x, int y) { return texelFetch(mask, ivec2(x, flip_mask == 1 ? mask_h - 1 - y : y), 0).x; }
void main() {
  vec2 uv = vec2(gl_FragCoord.x, out_size.y - gl_FragCoord.y) / out_size;   // canvas y points up
  vec2 size = vec2(mask_w, mask_h);
  vec2 s = clamp(uv * size - 0.5, vec2(0.0), size - 1.0);
  ivec2 p0 = ivec2(floor(s));
  ivec2 p1 = min(p0 + 1, ivec2(mask_w - 1, mask_h - 1));
  vec2 f = s - floor(s);
  float alpha = mix(mix(mask_at(p0.x, p0.y), mask_at(p1.x, p0.y), f.x),
                    mix(mask_at(p0.x, p1.y), mask_at(p1.x, p1.y), f.x), f.y);
  result = vec4(mix(color, texture(frame, uv).rgb, alpha), 1.0);
}`,
}

export class WebGLFrames {
  constructor(backend, canvas) {
    this.backend = backend
    this.gl = backend.gl
    this.canvas = canvas
    this.color = [0, 177 / 255, 64 / 255]   // green screen
    const gl = this.gl
    this.frameTexture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.preprocessProgram = backend.program('preprocess', 'none', SHADERS.preprocess)
    this.compositeProgram = backend.program('composite', 'none', SHADERS.composite)
  }

  setColor(rgb) {
    this.color = rgb
  }

  // Upload the frame. WebGL has no zero-copy import: for a VideoFrame this is
  // a GPU-side copy in Chrome, but it's still a copy.
  upload(frame) {
    const gl = this.gl
    const [width, height] = frameSize(frame)
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false)
    if (this.width === width && this.height === height) {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, frame)
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame)
      this.width = width
      this.height = height
    }
  }

  // Resize the uploaded frame into a vec4 tensor from the backend.
  preprocess(tensor) {
    const gl = this.gl
    this.backend.resetState()
    gl.useProgram(this.preprocessProgram)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture)
    gl.uniform1i(gl.getUniformLocation(this.preprocessProgram, 'frame'), 0)
    gl.uniform2f(gl.getUniformLocation(this.preprocessProgram, 'size'), tensor.w, tensor.h)
    this.backend.drawTo(tensor.fbo, tensor.w, tensor.h)
  }

  // Draw the uploaded frame over the background color into the canvas.
  composite(maskTexture, { width, height, flip = false }) {
    const gl = this.gl, p = this.compositeProgram
    if (this.canvas.width !== this.width || this.canvas.height !== this.height) {
      this.canvas.width = this.width
      this.canvas.height = this.height
    }
    this.backend.resetState()
    gl.useProgram(p)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.frameTexture)
    gl.uniform1i(gl.getUniformLocation(p, 'frame'), 0)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTexture)
    gl.uniform1i(gl.getUniformLocation(p, 'mask'), 1)
    gl.uniform2f(gl.getUniformLocation(p, 'out_size'), this.width, this.height)
    gl.uniform3fv(gl.getUniformLocation(p, 'color'), this.color)
    gl.uniform1i(gl.getUniformLocation(p, 'mask_w'), width)
    gl.uniform1i(gl.getUniformLocation(p, 'mask_h'), height)
    gl.uniform1i(gl.getUniformLocation(p, 'flip_mask'), flip ? 1 : 0)
    this.backend.drawTo(null, this.width, this.height)
  }

  output(frame) {
    return new VideoFrame(this.canvas, { timestamp: frame.timestamp, duration: frame.duration ?? undefined })
  }
}
