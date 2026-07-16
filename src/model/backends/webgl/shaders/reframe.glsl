#version 300 es
// Apply the reframe crop: sample `src` through the view rect. Same shape in and
// out. See reframe.wgsl for why this sits after the effect chain and touches
// only the foreground + alpha (never the background).
//
// rect = (cx, cy, size, moving), frame fractions; size ≤ 0 → identity.

precision highp float;
precision highp int;

uniform sampler2D u_src;
uniform sampler2D u_rect;   // 1×1
uniform int u_h;
uniform int u_w;

out vec4 fragColor;

vec4 samp(int x, int y) {
    return texelFetch(u_src, ivec2(clamp(x, 0, u_w - 1), clamp(y, 0, u_h - 1)), 0);
}

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec4 r = texelFetch(u_rect, ivec2(0, 0), 0);
    float s = r.z, rcx = r.x, rcy = r.y;
    if (s <= 0.0) { s = 1.0; rcx = 0.5; rcy = 0.5; }

    float fx = (rcx - s * 0.5) + ((float(x) + 0.5) / float(u_w)) * s;
    float fy = (rcy - s * 0.5) + ((float(y) + 0.5) / float(u_h)) * s;
    // fraction → texel index; an identity rect lands exactly on texel centres,
    // making the op a bit-exact copy.
    float sx = fx * float(u_w) - 0.5;
    float sy = fy * float(u_h) - 0.5;

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float tx = sx - float(x0);
    float ty = sy - float(y0);

    vec4 top = mix(samp(x0, y0),     samp(x0 + 1, y0),     tx);
    vec4 bot = mix(samp(x0, y0 + 1), samp(x0 + 1, y0 + 1), tx);
    fragColor = mix(top, bot, ty);
}
