#version 300 es
// proj_residual: bespoke 1×1 conv (no activation) + residual add, fused.
// Specializes conv2d_add to kernel=1 / stride=1 / pad=0 / no activation. Both
// inputs share the same spatial resolution. See conv2d.glsl for the weight
// texture layout (here K=1, so the weight texture is (inGroups*4, outGroups)).

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform sampler2D u_skip;
uniform sampler2D u_weights;
uniform sampler2D u_bias;

uniform int u_in_c_groups;
uniform int u_out_c_groups;

out vec4 fragColor;

void main() {
    int fx      = int(gl_FragCoord.x);
    int fy      = int(gl_FragCoord.y);
    int x_out   = fx / u_out_c_groups;
    int o_group = fx - x_out * u_out_c_groups;

    vec4 result = texelFetch(u_bias, ivec2(o_group, 0), 0);

    for (int i = 0; i < u_in_c_groups; i++) {
        vec4 in_val = texelFetch(u_input, ivec2(x_out * u_in_c_groups + i, fy), 0);

        int base = i * 4;
        mat4 w = mat4(
            texelFetch(u_weights, ivec2(base,     o_group), 0),
            texelFetch(u_weights, ivec2(base + 1, o_group), 0),
            texelFetch(u_weights, ivec2(base + 2, o_group), 0),
            texelFetch(u_weights, ivec2(base + 3, o_group), 0)
        );
        result += w * in_val;
    }

    fragColor = result + texelFetch(u_skip, ivec2(fx, fy), 0);
}
