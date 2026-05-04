#version 300 es
// Composite an RGBA image over a solid background color, gated by a
// 1-channel alpha mask. Output: premultiplied RGBA.
//
// Assumes image and alpha textures are the same h×w and that the canvas
// (viewport) matches that resolution — no resampling here. The upscaler op
// (separate) handles aligning alpha to the image resolution upstream.

precision highp float;

uniform sampler2D u_image;   // image as NHWC vec4 (RGBA in vec4)
uniform sampler2D u_alpha;   // alpha as NHWC vec4 (value in .r)
uniform vec3      u_bgColor; // straight-alpha background color, [0,1]

out vec4 fragColor;

void main() {
    // WebGL gl_FragCoord origin is bottom-left, but tensor textures are
    // stored top-down (matches NHWC + getImageData). Flip y when sampling so
    // the displayed image is upright. (WebGPU's @builtin(position) is already
    // top-down, so its compositor doesn't need this.)
    int H = textureSize(u_image, 0).y;
    ivec2 px = ivec2(int(gl_FragCoord.x), H - 1 - int(gl_FragCoord.y));
    vec3  fg = texelFetch(u_image, px, 0).rgb;
    float a  = texelFetch(u_alpha, px, 0).r;

    // Straight-alpha composite, then premultiply for the canvas's
    // premultiplied surface format.
    vec3 rgb = fg * a + u_bgColor * (1.0 - a);
    fragColor = vec4(rgb, 1.0);
}
