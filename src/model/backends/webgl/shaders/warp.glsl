#version 300 es
// Bilinear gather-warp. out[p] = sample(source, p + flow_scale·flow[p].xy),
// edge-clamped. Source/output are (W·G, H) with G = c/4 channel groups
// (fragment x decomposes to pixel + group, like conv); flow is 4-ch, same
// spatial resolution, vector in .xy.

precision highp float;
precision highp int;

uniform sampler2D u_source;   // (W * G, H)
uniform sampler2D u_flow;     // (W, H), flow in .xy
uniform int   u_w;
uniform int   u_h;
uniform int   u_groups;
uniform float u_flow_scale;

out vec4 fragColor;

vec4 samp(int x, int y, int g) {
    int cx = clamp(x, 0, u_w - 1);
    int cy = clamp(y, 0, u_h - 1);
    return texelFetch(u_source, ivec2(cx * u_groups + g, cy), 0);
}

void main() {
    int fx = int(gl_FragCoord.x);
    int y  = int(gl_FragCoord.y);
    int x  = fx / u_groups;
    int g  = fx - x * u_groups;

    vec2 f  = texelFetch(u_flow, ivec2(x, y), 0).xy;
    float sx = clamp(float(x) + u_flow_scale * f.x, 0.0, float(u_w - 1));
    float sy = clamp(float(y) + u_flow_scale * f.y, 0.0, float(u_h - 1));

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float tx = sx - float(x0);
    float ty = sy - float(y0);

    vec4 top = mix(samp(x0, y0, g), samp(x0 + 1, y0, g), tx);
    vec4 bot = mix(samp(x0, y0 + 1, g), samp(x0 + 1, y0 + 1, g), tx);
    fragColor = mix(top, bot, ty);
}
