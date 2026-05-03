#version 300 es
// Element-wise sigmoid — works on any texture layout (flat or spatial).

precision highp float;

uniform sampler2D u_input;

out vec4 fragColor;

void main() {
    vec4 x = texelFetch(u_input, ivec2(gl_FragCoord.xy), 0);
    fragColor = 1.0 / (1.0 + exp(-x));
}
