#version 300 es
// up_final_skip (C/D alpha head): cat(u[2], d_full[4], rgb[3]) → conv 9→1 →
// sigmoid. Weights = 27 vec4 (3 per kpos): [kpos*3+0]=(w0,w1,0,0) u;
// [kpos*3+1]=(w2..w5) d_full; [kpos*3+2]=(w6,w7,w8,0) rgb.

precision highp float;
precision highp int;

uniform sampler2D u_u_gru;    // (W, H) — .xy
uniform sampler2D u_d_full;   // (W, H) — full vec4
uniform sampler2D u_rgb;      // (W, H) — .xyz
uniform sampler2D u_weights;  // (27, 1)
uniform sampler2D u_bias;     // (1, 1) — .x
uniform int u_w;
uniform int u_h;

out vec4 fragColor;

float wf(int i) { return texelFetch(u_weights, ivec2(i / 4, 0), 0)[i & 3]; }

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
            vec4 u = texelFetch(u_u_gru,  ivec2(ix, iy), 0);
            vec4 d = texelFetch(u_d_full, ivec2(ix, iy), 0);
            vec4 r = texelFetch(u_rgb,    ivec2(ix, iy), 0);
            int b0 = (kpos * 3 + 0) * 4, b1 = (kpos * 3 + 1) * 4, b2 = (kpos * 3 + 2) * 4;
            acc += wf(b0 + 0) * u.x + wf(b0 + 1) * u.y;
            acc += wf(b1 + 0) * d.x + wf(b1 + 1) * d.y + wf(b1 + 2) * d.z + wf(b1 + 3) * d.w;
            acc += wf(b2 + 0) * r.x + wf(b2 + 1) * r.y + wf(b2 + 2) * r.z;
        }
    }

    fragColor = vec4(1.0 / (1.0 + exp(-acc)), 0.0, 0.0, 0.0);
}
