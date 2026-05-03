#version 300 es
// Bilinear upsample (arbitrary scale) — align_corners=False.
//
// Hardware texture filtering can't be used: adjacent texels in x belong to
// different channel groups, not spatial neighbours. Manual 4-tap blend.
//
// Input:  (in_W  * c_groups, in_H)
// Output: (out_W * c_groups, out_H)

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform int u_in_w;
uniform int u_in_h;
uniform int u_out_w;
uniform int u_out_h;
uniform int u_c_groups;

out vec4 fragColor;

void main() {
    ivec2 fc    = ivec2(gl_FragCoord.xy);
    int x_out   = fc.x / u_c_groups;
    int y_out   = fc.y;
    int c_group = fc.x - x_out * u_c_groups;

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

    vec4 v00 = texelFetch(u_input, ivec2(x0 * u_c_groups + c_group, y0), 0);
    vec4 v10 = texelFetch(u_input, ivec2(x1 * u_c_groups + c_group, y0), 0);
    vec4 v01 = texelFetch(u_input, ivec2(x0 * u_c_groups + c_group, y1), 0);
    vec4 v11 = texelFetch(u_input, ivec2(x1 * u_c_groups + c_group, y1), 0);

    fragColor = mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}
