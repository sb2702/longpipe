#version 300 es
// gates_fused: ConvGRU z + r gates, fused. Production config (c_up=2,
// recurrent=1). u_in (.x=a, .y=b), h_prev (.x). weights 9 vec4 =
// (z_w_b, z_w_h, r_w_b, r_w_h) in a (9,1) texture. bias .xy. output (z, r, 0, 0).

precision highp float;
precision highp int;

uniform sampler2D u_u_in;     // (W, H)
uniform sampler2D u_h_prev;   // (W, H)
uniform sampler2D u_weights;  // (9, 1)
uniform sampler2D u_bias;     // (1, 1)
uniform int u_w;
uniform int u_h;

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec2 b = texelFetch(u_bias, ivec2(0, 0), 0).xy;
    float z_pre = b.x;
    float r_pre = b.y;

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int iy = y + ky - 1;
            int ix = x + kx - 1;
            if (iy < 0 || ix < 0 || iy >= u_h || ix >= u_w) continue;
            int kpos  = ky * 3 + kx;
            float b_n = texelFetch(u_u_in,   ivec2(ix, iy), 0).y;
            float h_n = texelFetch(u_h_prev, ivec2(ix, iy), 0).z;
            vec4 w    = texelFetch(u_weights, ivec2(kpos, 0), 0);
            z_pre += w.x * b_n + w.y * h_n;
            r_pre += w.z * b_n + w.w * h_n;
        }
    }

    float z = 1.0 / (1.0 + exp(-z_pre));
    float r = 1.0 / (1.0 + exp(-r_pre));
    fragColor = vec4(z, r, 0.0, 0.0);
}
