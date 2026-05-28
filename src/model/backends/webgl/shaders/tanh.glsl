#version 300 es
// Element-wise tanh — used by ConvGRU candidate activation.
// Layout-agnostic (flat or spatial); GLSL tanh is element-wise on vec4.

precision highp float;

uniform sampler2D u_input;

out vec4 fragColor;

void main() {
    fragColor = tanh(texelFetch(u_input, ivec2(gl_FragCoord.xy), 0));
}
