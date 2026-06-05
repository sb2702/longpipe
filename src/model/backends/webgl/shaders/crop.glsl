#version 300 es
// Top-left crop: the output viewport is smaller than the input texture, so reading
// the same fragment coord yields the top-left subregion (channel groups packed in
// the x axis line up because group count is unchanged).

precision highp float;
precision highp int;

uniform sampler2D u_input;
out vec4 fragColor;

void main() {
    fragColor = texelFetch(u_input, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0);
}
