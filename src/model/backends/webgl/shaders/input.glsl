#version 300 es
precision highp float;

// Input op — bilinear-sample a source texture (RGBA8 unorm uploaded from an
// ImageBitmap or VideoFrame) at the target resolution. Output texture format
// is RGBA32F or RGBA16F depending on backend.dtype; the framebuffer write
// stores fragColor into whatever format is bound.

uniform sampler2D u_src;
uniform int       u_out_w;
uniform int       u_out_h;
// Orientation: upright uv → source uv (see model/orientation.ts).
uniform vec3      u_uv_x;
uniform vec3      u_uv_y;

out vec4 fragColor;

void main() {
    // gl_FragCoord is already at the pixel centre (x.5).
    vec3 o = vec3(gl_FragCoord.xy / vec2(float(u_out_w), float(u_out_h)), 1.0);
    fragColor = texture(u_src, vec2(dot(u_uv_x, o), dot(u_uv_y, o)));
}
