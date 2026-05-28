#version 300 es
// Element-wise multiply — used in ConvGRU for r ⊙ h_prev.
// Same shape constraint as add.glsl.

precision highp float;

uniform sampler2D u_input_a;
uniform sampler2D u_input_b;

out vec4 fragColor;

void main() {
    ivec2 fc = ivec2(gl_FragCoord.xy);
    fragColor = texelFetch(u_input_a, fc, 0) * texelFetch(u_input_b, fc, 0);
}
