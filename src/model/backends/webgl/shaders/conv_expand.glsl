#version 300 es
// conv_expand: bespoke N→2 conv 3×3 (pad 1) + relu (wrapper expand_feat).
// Input N ch (in_groups vec4); output 2ch in .xy (.zw = 0). Weights: 9 *
// in_groups mat4x2 (8 floats each, col-major c0r0,c0r1,...,c3r0,c3r1) in a
// 1-row texture — bounded small (feat_ch ≤ 32).

precision highp float;
precision highp int;

uniform sampler2D u_input;    // (W*in_groups, H)
uniform sampler2D u_weights;  // (ceil(9*in_groups*8/4), 1)
uniform sampler2D u_bias;     // (1, 1) — .xy
uniform int u_w;
uniform int u_h;
uniform int u_in_groups;

out vec4 fragColor;

float wf(int i) { return texelFetch(u_weights, ivec2(i / 4, 0), 0)[i & 3]; }

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec2 result = texelFetch(u_bias, ivec2(0, 0), 0).xy;

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int iy = y + ky - 1;
            int ix = x + kx - 1;
            if (iy < 0 || ix < 0 || iy >= u_h || ix >= u_w) continue;
            int kpos = ky * 3 + kx;
            for (int ig = 0; ig < u_in_groups; ig++) {
                vec4 v = texelFetch(u_input, ivec2(ix * u_in_groups + ig, iy), 0);
                int base = (kpos * u_in_groups + ig) * 8;  // mat4x2 = 8 floats
                result.x += wf(base + 0) * v.x + wf(base + 2) * v.y + wf(base + 4) * v.z + wf(base + 6) * v.w;
                result.y += wf(base + 1) * v.x + wf(base + 3) * v.y + wf(base + 5) * v.z + wf(base + 7) * v.w;
            }
        }
    }

    fragColor = vec4(max(result, vec2(0.0)), 0.0, 0.0);   // expand_feat is F.relu
}
