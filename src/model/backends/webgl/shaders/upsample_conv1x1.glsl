#version 300 es
// Bilinear upsample + 1×1 pointwise conv fused.
// For each output pixel + out group, bilinearly samples each in_group from input,
// applies the 1×1 conv weight, and writes the activated result.
//
// Input:   (in_W  * in_groups,  in_H)
// Output:  (out_W * out_groups, out_H)
// Weights: (in_groups * 4, out_groups)  (K=1, so kernel-row dimension collapses)
//          mat4[col][row] = weight(in_channel=col, out_channel=row)
// Bias:    (out_groups, 1)

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform sampler2D u_weights;
uniform sampler2D u_bias;

uniform int u_in_w;
uniform int u_in_h;
uniform int u_out_w;
uniform int u_out_h;
uniform int u_in_c_groups;
uniform int u_out_c_groups;
uniform int u_activation;   // 0 = none, 1 = relu6, 2 = relu

out vec4 fragColor;

void main() {
    int fx      = int(gl_FragCoord.x);
    int fy      = int(gl_FragCoord.y);
    int x_out   = fx / u_out_c_groups;
    int y_out   = fy;
    int o_group = fx - x_out * u_out_c_groups;

    float scale_x = float(u_in_w) / float(u_out_w);
    float scale_y = float(u_in_h) / float(u_out_h);
    float src_x = (float(x_out) + 0.5) * scale_x - 0.5;
    float src_y = (float(y_out) + 0.5) * scale_y - 0.5;

    int x0 = int(floor(src_x));
    int y0 = int(floor(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx_w = src_x - float(x0);
    float fy_w = src_y - float(y0);

    x0 = clamp(x0, 0, u_in_w - 1);
    y0 = clamp(y0, 0, u_in_h - 1);
    x1 = clamp(x1, 0, u_in_w - 1);
    y1 = clamp(y1, 0, u_in_h - 1);

    vec4 result = texelFetch(u_bias, ivec2(o_group, 0), 0);

    for (int i = 0; i < u_in_c_groups; i++) {
        vec4 v00 = texelFetch(u_input, ivec2(x0 * u_in_c_groups + i, y0), 0);
        vec4 v10 = texelFetch(u_input, ivec2(x1 * u_in_c_groups + i, y0), 0);
        vec4 v01 = texelFetch(u_input, ivec2(x0 * u_in_c_groups + i, y1), 0);
        vec4 v11 = texelFetch(u_input, ivec2(x1 * u_in_c_groups + i, y1), 0);

        vec4 sampled = mix(mix(v00, v10, fx_w), mix(v01, v11, fx_w), fy_w);

        int base = i * 4;
        mat4 w = mat4(
            texelFetch(u_weights, ivec2(base,     o_group), 0),
            texelFetch(u_weights, ivec2(base + 1, o_group), 0),
            texelFetch(u_weights, ivec2(base + 2, o_group), 0),
            texelFetch(u_weights, ivec2(base + 3, o_group), 0)
        );
        result += w * sampled;
    }

    if      (u_activation == 1) result = clamp(result, 0.0, 6.0);
    else if (u_activation == 2) result = max(result, vec4(0.0));

    fragColor = result;
}
