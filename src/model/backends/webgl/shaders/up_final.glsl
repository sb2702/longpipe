#version 300 es
// up_final: cat(u[2], rgb[3]) → conv 3×3 5→1 → sigmoid (A/B alpha head).
// weights 18 vec4 (1-row): [kpos]=(w0,w1,0,0) for u, [9+kpos]=(w2,w3,w4,0) rgb.

precision highp float;
precision highp int;

uniform sampler2D u_u_gru;    // (W, H) — .xy
uniform sampler2D u_rgb;      // (W, H) — .xyz
uniform sampler2D u_weights;  // (18, 1)
uniform sampler2D u_bias;     // (1, 1) — .x
uniform int u_w;
uniform int u_h;

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    float acc = texelFetch(u_bias, ivec2(0, 0), 0).x;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int iy = y + ky - 1;
            int ix = x + kx - 1;
            if (iy < 0 || ix < 0 || iy >= u_h || ix >= u_w) continue;
            int kpos = ky * 3 + kx;
            vec4 u  = texelFetch(u_u_gru,   ivec2(ix, iy), 0);
            vec4 r  = texelFetch(u_rgb,     ivec2(ix, iy), 0);
            vec4 wu = texelFetch(u_weights, ivec2(kpos, 0), 0);
            vec4 wr = texelFetch(u_weights, ivec2(9 + kpos, 0), 0);
            acc += dot(wu.xy, u.xy);
            acc += dot(wr.xyz, r.xyz);
        }
    }

    fragColor = vec4(1.0 / (1.0 + exp(-acc)), 0.0, 0.0, 0.0);
}
