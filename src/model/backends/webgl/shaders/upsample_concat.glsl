#version 300 es
// Bilinear upsample (input_a) + channel concat (with input_b) fused.
// input_a is the decoder tensor at small spatial res (in_h × in_w), bilinearly resized to (out_h × out_w).
// input_b is the encoder skip feature already at output resolution.
// Output channels [0..a_groups-1] come from upsampled input_a.
// Output channels [a_groups..out_groups-1] come from input_b (passthrough).

precision highp float;
precision highp int;

uniform sampler2D u_input_a;
uniform sampler2D u_input_b;
uniform int u_in_w;
uniform int u_in_h;
uniform int u_out_w;
uniform int u_out_h;
uniform int u_a_c_groups;
uniform int u_b_c_groups;

out vec4 fragColor;

void main() {
    ivec2 fc       = ivec2(gl_FragCoord.xy);
    int out_groups = u_a_c_groups + u_b_c_groups;
    int x_out      = fc.x / out_groups;
    int c_out      = fc.x - x_out * out_groups;
    int y_out      = fc.y;

    if (c_out < u_a_c_groups) {
        // Bilinear sample from input_a at (x_out, y_out) for channel group c_out.
        float scale_x = float(u_in_w) / float(u_out_w);
        float scale_y = float(u_in_h) / float(u_out_h);
        float src_x = (float(x_out) + 0.5) * scale_x - 0.5;
        float src_y = (float(y_out) + 0.5) * scale_y - 0.5;

        int x0 = int(floor(src_x));
        int y0 = int(floor(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = src_x - float(x0);
        float fy = src_y - float(y0);

        x0 = clamp(x0, 0, u_in_w - 1);
        y0 = clamp(y0, 0, u_in_h - 1);
        x1 = clamp(x1, 0, u_in_w - 1);
        y1 = clamp(y1, 0, u_in_h - 1);

        vec4 v00 = texelFetch(u_input_a, ivec2(x0 * u_a_c_groups + c_out, y0), 0);
        vec4 v10 = texelFetch(u_input_a, ivec2(x1 * u_a_c_groups + c_out, y0), 0);
        vec4 v01 = texelFetch(u_input_a, ivec2(x0 * u_a_c_groups + c_out, y1), 0);
        vec4 v11 = texelFetch(u_input_a, ivec2(x1 * u_a_c_groups + c_out, y1), 0);

        fragColor = mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
    } else {
        int c_b = c_out - u_a_c_groups;
        fragColor = texelFetch(u_input_b, ivec2(x_out * u_b_c_groups + c_b, y_out), 0);
    }
}
