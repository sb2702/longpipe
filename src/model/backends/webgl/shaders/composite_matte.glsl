#version 300 es
// Render the raw 1-channel alpha matte as a premultiplied white silhouette
// (rgb = a, alpha = a). Doubles as a debug view AND a reusable mask: a consumer
// can composite it against their own full-resolution source (e.g. drawImage
// with globalCompositeOperation 'destination-in'). No image input — matte only.
//
// Assumes the canvas (viewport) matches the alpha texture h×w — no resampling.

precision highp float;

uniform sampler2D u_alpha;   // alpha as NHWC vec4 (value in .r)

out vec4 fragColor;

void main() {
    // WebGL gl_FragCoord origin is bottom-left; tensor textures are stored
    // top-down. Flip y so the displayed mask is upright.
    int H = textureSize(u_alpha, 0).y;
    ivec2 px = ivec2(int(gl_FragCoord.x), H - 1 - int(gl_FragCoord.y));
    float a = texelFetch(u_alpha, px, 0).r;
    // Premultiplied white × matte: (a, a, a, a).
    fragColor = vec4(vec3(a), a);
}
