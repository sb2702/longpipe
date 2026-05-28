#version 300 es
// Per-channel scaled residual: b_out = b + γ ⊙ h_new.
// γ is one f32 per channel laid out as (c_groups, 1) — one vec4 per group.

precision highp float;
precision highp int;

uniform sampler2D u_b;
uniform sampler2D u_h_new;
uniform sampler2D u_gamma;     // (c_groups, 1)
uniform int u_c_groups;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    int x_spatial = fc.x / u_c_groups;
    int c         = fc.x - x_spatial * u_c_groups;

    vec4 b     = texelFetch(u_b,     fc, 0);
    vec4 h_new = texelFetch(u_h_new, fc, 0);
    vec4 gamma = texelFetch(u_gamma, ivec2(c, 0), 0);

    fragColor = b + gamma * h_new;
}
