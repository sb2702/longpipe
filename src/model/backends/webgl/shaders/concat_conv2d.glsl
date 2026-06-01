#version 300 es
// concat_conv2d: fuses [concat(a, b) → conv 3×3 (pad 1) → relu6] into one pass.
// Both inputs already at output resolution. Weight cols [0, a_groups) read a,
// [a_groups, I) read b. Weight texture layout matches conv2d.glsl
// ((in_groups*4, 9*out_groups), in_groups = a_groups + b_groups).

precision highp float;
precision highp int;

uniform sampler2D u_a;
uniform sampler2D u_b;
uniform sampler2D u_weights;
uniform sampler2D u_bias;

uniform int u_w;
uniform int u_h;
uniform int u_a_groups;
uniform int u_b_groups;
uniform int u_out_c_groups;

out vec4 fragColor;

void main() {
    int fx      = int(gl_FragCoord.x);
    int fy      = int(gl_FragCoord.y);
    int x_out   = fx / u_out_c_groups;
    int o_group = fx - x_out * u_out_c_groups;
    int A = u_a_groups;
    int B = u_b_groups;
    int I = A + B;

    vec4 result = texelFetch(u_bias, ivec2(o_group, 0), 0);

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int in_y = fy    + ky - 1;
            int in_x = x_out + kx - 1;
            if (in_x < 0 || in_y < 0 || in_x >= u_w || in_y >= u_h) continue;

            int z     = ky * 3 + kx;
            int w_row = z * u_out_c_groups + o_group;

            for (int i = 0; i < I; i++) {
                vec4 in_val = (i < A)
                    ? texelFetch(u_a, ivec2(in_x * A + i,       in_y), 0)
                    : texelFetch(u_b, ivec2(in_x * B + (i - A), in_y), 0);

                int base = i * 4;
                mat4 w = mat4(
                    texelFetch(u_weights, ivec2(base,     w_row), 0),
                    texelFetch(u_weights, ivec2(base + 1, w_row), 0),
                    texelFetch(u_weights, ivec2(base + 2, w_row), 0),
                    texelFetch(u_weights, ivec2(base + 3, w_row), 0)
                );
                result += w * in_val;
            }
        }
    }

    fragColor = clamp(result, 0.0, 6.0);
}
