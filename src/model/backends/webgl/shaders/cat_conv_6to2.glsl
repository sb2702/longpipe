#version 300 es
// cat_conv_6to2: fused concat(u[2], d[4]) → 6→2 conv 3×3 (pad 1) + relu
// (E up1_combine). Channel order v3a=(u.x,u.y,d.x), v3b=d.yzw. weights = 9*2
// mat3x2 (6 floats each, col-major) in a 1-row (27-texel) texture.

precision highp float;
precision highp int;

uniform sampler2D u_u_in;     // (W, H) — .xy
uniform sampler2D u_d_in;     // (W, H) — full vec4
uniform sampler2D u_weights;  // (27, 1)
uniform sampler2D u_bias;     // (1, 1) — .xy
uniform int u_w;
uniform int u_h;

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
            vec4 u = texelFetch(u_u_in, ivec2(ix, iy), 0);
            vec4 d = texelFetch(u_d_in, ivec2(ix, iy), 0);
            vec3 v3a = vec3(u.x, u.y, d.x);
            vec3 v3b = d.yzw;
            for (int ig = 0; ig < 2; ig++) {
                int base = (kpos * 2 + ig) * 6;
                vec3 v3 = (ig == 0) ? v3a : v3b;
                for (int col = 0; col < 3; col++) {
                    result.x += wf(base + col * 2 + 0) * v3[col];
                    result.y += wf(base + col * 2 + 1) * v3[col];
                }
            }
        }
    }

    fragColor = vec4(max(result, vec2(0.0)), 0.0, 0.0);   // up1_combine is F.relu
}
