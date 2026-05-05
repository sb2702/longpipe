#version 300 es
// Like composite_image but bg is bilinearly sampled — bg is at a smaller
// resolution than (image, alpha) at canvas h × w. Used by CompositorBlur to
// skip the final full-res upsample in the blur pyramid.
//
// Tensor textures are top-down (origin at top-left of the source image),
// while gl_FragCoord origin is bottom-left — flip y when computing px.

precision highp float;
precision highp int;

uniform sampler2D u_image;
uniform sampler2D u_alpha;
uniform sampler2D u_bg;

uniform int u_out_w;
uniform int u_out_h;
uniform int u_bg_w;
uniform int u_bg_h;

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int y = u_out_h - 1 - int(gl_FragCoord.y);

    vec3  fg = texelFetch(u_image, ivec2(x, y), 0).rgb;
    float a  = texelFetch(u_alpha, ivec2(x, y), 0).r;

    // Bilinear sample bg at the corresponding location. align_corners=False.
    float src_x = (float(x) + 0.5) * (float(u_bg_w) / float(u_out_w)) - 0.5;
    float src_y = (float(y) + 0.5) * (float(u_bg_h) / float(u_out_h)) - 0.5;
    int x0 = clamp(int(floor(src_x)),     0, u_bg_w - 1);
    int x1 = clamp(int(floor(src_x)) + 1, 0, u_bg_w - 1);
    int y0 = clamp(int(floor(src_y)),     0, u_bg_h - 1);
    int y1 = clamp(int(floor(src_y)) + 1, 0, u_bg_h - 1);
    float wx = src_x - floor(src_x);
    float wy = src_y - floor(src_y);

    vec3 tl = texelFetch(u_bg, ivec2(x0, y0), 0).rgb;
    vec3 tr = texelFetch(u_bg, ivec2(x1, y0), 0).rgb;
    vec3 bl = texelFetch(u_bg, ivec2(x0, y1), 0).rgb;
    vec3 br = texelFetch(u_bg, ivec2(x1, y1), 0).rgb;
    vec3 bg = (1.0 - wy) * ((1.0 - wx) * tl + wx * tr)
            +        wy  * ((1.0 - wx) * bl + wx * br);

    vec3 rgb = fg * a + bg * (1.0 - a);
    fragColor = vec4(rgb, 1.0);
}
