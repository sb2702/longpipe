#version 300 es
// Passthrough "compositor": writes the image directly to the canvas
// (default framebuffer). No alpha math, no background. Used by RenderOp
// when the renderer is in disabled state.
//
// Caller invariants:
//   - canvas (viewport) === image h × w (no resampling here)

precision highp float;

uniform sampler2D u_image;   // image as NHWC vec4 (RGBA)

out vec4 fragColor;

void main() {
    // WebGL gl_FragCoord origin is bottom-left; tensor textures are stored
    // top-down. Flip y so the displayed image is upright. (Matches what the
    // existing composite_solid.glsl does.)
    int H = textureSize(u_image, 0).y;
    ivec2 px = ivec2(int(gl_FragCoord.x), H - 1 - int(gl_FragCoord.y));
    vec3  rgb = texelFetch(u_image, px, 0).rgb;
    fragColor = vec4(rgb, 1.0);
}
