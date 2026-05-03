#version 300 es
// Channel concat — concatenates two NHWC textures along the channel dimension.
// Both inputs must share the same spatial dimensions (W, H). Channel counts are
// multiples of 4 so each "group" is one vec4 texel.
//
// Input  A: (W * a_c_groups, H)
// Input  B: (W * b_c_groups, H)
// Output:   (W * (a_c_groups + b_c_groups), H)

precision highp float;
precision highp int;

uniform sampler2D u_input_a;
uniform sampler2D u_input_b;
uniform int u_a_c_groups;
uniform int u_b_c_groups;

out vec4 fragColor;

void main() {
    ivec2 fc       = ivec2(gl_FragCoord.xy);
    int out_groups = u_a_c_groups + u_b_c_groups;
    int x_spatial  = fc.x / out_groups;
    int c_out      = fc.x - x_spatial * out_groups;
    int y          = fc.y;

    if (c_out < u_a_c_groups) {
        fragColor = texelFetch(u_input_a, ivec2(x_spatial * u_a_c_groups + c_out, y), 0);
    } else {
        int c_b = c_out - u_a_c_groups;
        fragColor = texelFetch(u_input_b, ivec2(x_spatial * u_b_c_groups + c_b, y), 0);
    }
}
