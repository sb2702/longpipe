#version 300 es
precision highp float;

// Input op — bilinear-sample a source texture (RGBA8 unorm uploaded from an
// ImageBitmap or VideoFrame) at the target resolution. Output texture format
// is RGBA32F or RGBA16F depending on backend.dtype; the framebuffer write
// stores fragColor into whatever format is bound.

uniform sampler2D u_src;
uniform int       u_out_w;
uniform int       u_out_h;

out vec4 fragColor;

void main() {
    vec2 uv = (gl_FragCoord.xy + vec2(0.5)) / vec2(float(u_out_w), float(u_out_h));
    fragColor = texture(u_src, uv);
}
