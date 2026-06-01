#version 300 es
// cand_update_fused: ConvGRU candidate + state update + output, fused.
// Production config (c_up=2, recurrent=1). u_in (.x=a, .y=b), h_prev (.x),
// gates_out (.x=z, .y=r). weights 9 vec4 (.xy = b_w, rh_w) in a (9,1) texture.
// bias .x, gamma .x. output (a, b_out, 0, 0).

precision highp float;
precision highp int;

uniform sampler2D u_u_in;       // (W, H)
uniform sampler2D u_h_prev;     // (W, H)
uniform sampler2D u_gates_out;  // (W, H)
uniform sampler2D u_weights;    // (9, 1)
uniform sampler2D u_bias;       // (1, 1)
uniform sampler2D u_gamma;      // (1, 1)
uniform int u_w;
uniform int u_h;

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    float cand_pre = texelFetch(u_bias, ivec2(0, 0), 0).x;
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int iy = y + ky - 1;
            int ix = x + kx - 1;
            if (iy < 0 || ix < 0 || iy >= u_h || ix >= u_w) continue;
            int kpos  = ky * 3 + kx;
            float b_n = texelFetch(u_u_in,      ivec2(ix, iy), 0).y;
            float h_n = texelFetch(u_h_prev,    ivec2(ix, iy), 0).z;
            float r_n = texelFetch(u_gates_out, ivec2(ix, iy), 0).y;
            vec4 w    = texelFetch(u_weights,   ivec2(kpos, 0), 0);
            cand_pre += w.x * b_n + w.y * (r_n * h_n);
        }
    }

    float h_til      = tanh(cand_pre);
    vec4  u_cur      = texelFetch(u_u_in,      ivec2(x, y), 0);
    float z_cur      = texelFetch(u_gates_out, ivec2(x, y), 0).x;
    float h_prev_cur = texelFetch(u_h_prev,    ivec2(x, y), 0).z;
    float h_new      = (1.0 - z_cur) * h_prev_cur + z_cur * h_til;
    float gamma      = texelFetch(u_gamma,     ivec2(0, 0), 0).x;
    float b_out      = u_cur.y + gamma * h_new;
    fragColor = vec4(u_cur.x, b_out, h_new, 0.0);
}
