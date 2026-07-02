#version 300 es
// Composite an RGBA image over TRANSPARENCY, gated by a 1-channel alpha mask.
// Output: premultiplied RGBA with the matte as the alpha channel, so the
// subject is isolated on a transparent background — whatever sits behind the
// canvas shows through wherever the matte is 0. Mirrors composite_solid.glsl
// but drops the background color (the "background" is nothing).
//
// Assumes image and alpha textures are the same h×w and that the canvas
// (viewport) matches that resolution — no resampling here.

precision highp float;

uniform sampler2D u_image;   // image as NHWC vec4 (RGBA in vec4)
uniform sampler2D u_alpha;   // alpha as NHWC vec4 (value in .r)

out vec4 fragColor;

void main() {
    // WebGL gl_FragCoord origin is bottom-left, but tensor textures are stored
    // top-down. Flip y when sampling so the displayed image is upright (matches
    // composite_solid.glsl).
    int H = textureSize(u_image, 0).y;
    ivec2 px = ivec2(int(gl_FragCoord.x), H - 1 - int(gl_FragCoord.y));
    vec3  fg = texelFetch(u_image, px, 0).rgb;
    float a  = texelFetch(u_alpha, px, 0).r;

    // Premultiplied output: rgb·a with the matte as the alpha channel. On the
    // canvas's premultiplied surface this is a correct straight-alpha subject
    // over transparency.
    fragColor = vec4(fg * a, a);
}
