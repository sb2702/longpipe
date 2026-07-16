#version 300 es
// Box-driven square crop + bilinear resample + ((rgb - mean)/std) normalize.
// Reads the box tensor (face_box output: cx, cy, halfSide, score in frame
// fractions; halfSide as a fraction of frame WIDTH) and emits the landmark
// model's input, .w = 0. Sampling mirrors landmark training's warpAffine:
// src = center + (u - out/2) · side/out.
//
// frame texture: (w, h) — 4-ch, 1 group. box texture: 1×1.

precision highp float;
precision highp int;

uniform sampler2D u_frame;
uniform sampler2D u_box;
uniform int u_in_h;
uniform int u_in_w;
uniform int u_out_h;
uniform int u_out_w;
uniform int u_slot;   // box-tensor slot (multi-face); 0 for the single-face path
uniform float u_mean_r;
uniform float u_mean_g;
uniform float u_mean_b;
uniform float u_std_r;
uniform float u_std_g;
uniform float u_std_b;

out vec4 fragColor;

vec4 samp(int x, int y) {
    return texelFetch(u_frame, ivec2(clamp(x, 0, u_in_w - 1), clamp(y, 0, u_in_h - 1)), 0);
}

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec4 box = texelFetch(u_box, ivec2(u_slot, 0), 0);
    float cx   = box.x * float(u_in_w);
    float cy   = box.y * float(u_in_h);
    float side = 2.0 * box.z * float(u_in_w);

    float sx = clamp(cx + (float(x) - float(u_out_w) * 0.5) * side / float(u_out_w), 0.0, float(u_in_w - 1));
    float sy = clamp(cy + (float(y) - float(u_out_h) * 0.5) * side / float(u_out_h), 0.0, float(u_in_h - 1));

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float tx = sx - float(x0);
    float ty = sy - float(y0);

    vec4 top = mix(samp(x0, y0),     samp(x0 + 1, y0),     tx);
    vec4 bot = mix(samp(x0, y0 + 1), samp(x0 + 1, y0 + 1), tx);
    vec4 v = mix(top, bot, ty);

    vec3 mean = vec3(u_mean_r, u_mean_g, u_mean_b);
    vec3 std  = vec3(u_std_r, u_std_g, u_std_b);
    fragColor = vec4((v.rgb - mean) / std, 0.0);
}
