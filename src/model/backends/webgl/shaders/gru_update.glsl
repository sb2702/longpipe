#version 300 es
// Fused ConvGRU state update: h_new = (1 - z) ⊙ h_prev + z ⊙ h_til.
// All three inputs share shape; layout-agnostic.

precision highp float;

uniform sampler2D u_z;
uniform sampler2D u_h_prev;
uniform sampler2D u_h_til;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    vec4 z   = texelFetch(u_z,      fc, 0);
    vec4 h_p = texelFetch(u_h_prev, fc, 0);
    vec4 h_t = texelFetch(u_h_til,  fc, 0);
    fragColor = (vec4(1.0) - z) * h_p + z * h_t;
}
