#version 300 es
// Per-channel scale: x (W * C_groups, H) * scale (C_groups, 1), broadcast over H×W.
precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform sampler2D u_scale;
uniform int u_c_groups;

out vec4 fragColor;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    fragColor = texelFetch(u_input, p, 0) * texelFetch(u_scale, ivec2(p.x % u_c_groups, 0), 0);
}
