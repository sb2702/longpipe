#version 300 es
// Conv2d + skip add fused — identical to conv2d.glsl with a skip texture added at write time.
// See conv2d.glsl for tensor/weight layout details.

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform sampler2D u_skip;
uniform sampler2D u_weights;
uniform sampler2D u_bias;

uniform int u_in_w;
uniform int u_in_h;
uniform int u_in_c_groups;
uniform int u_out_c_groups;
uniform int u_kernel_h;
uniform int u_kernel_w;
uniform int u_stride;
uniform int u_pad_top;
uniform int u_pad_left;
uniform int u_activation;   // 0 = none, 1 = relu6, 2 = relu

out vec4 fragColor;

void main() {
    int fx      = int(gl_FragCoord.x);
    int fy      = int(gl_FragCoord.y);
    int x_out   = fx / u_out_c_groups;
    int y_out   = fy;
    int o_group = fx - x_out * u_out_c_groups;

    vec4 result = texelFetch(u_bias, ivec2(o_group, 0), 0);

    for (int ky = 0; ky < u_kernel_h; ky++) {
        for (int kx = 0; kx < u_kernel_w; kx++) {
            int in_y = y_out * u_stride + ky - u_pad_top;
            int in_x = x_out * u_stride + kx - u_pad_left;

            if (in_y < 0 || in_y >= u_in_h || in_x < 0 || in_x >= u_in_w) continue;

            int z     = ky * u_kernel_w + kx;
            int w_row = z * u_out_c_groups + o_group;

            for (int i = 0; i < u_in_c_groups; i++) {
                vec4 in_val = texelFetch(u_input, ivec2(in_x * u_in_c_groups + i, in_y), 0);

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

    if      (u_activation == 1) result = clamp(result, 0.0, 6.0);
    else if (u_activation == 2) result = max(result, vec4(0.0));

    fragColor = result + texelFetch(u_skip, ivec2(fx, fy), 0);
}
