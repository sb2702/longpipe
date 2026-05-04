#version 300 es
// Separable 1D Gaussian blur — run twice (h then v) for a 2D blur.
// Compile-time radius; sigma is a runtime uniform.

precision highp float;
precision highp int;

uniform sampler2D u_input;
uniform ivec2     u_step;     // (1,0) for horizontal pass, (0,1) for vertical
uniform float     u_sigma;
uniform int       u_in_w;
uniform int       u_in_h;
uniform int       u_c_groups;

const int R = 32;

out vec4 fragColor;

void main() {
    ivec2 fc    = ivec2(gl_FragCoord.xy);
    int x       = fc.x / u_c_groups;
    int c_group = fc.x - x * u_c_groups;
    int y       = fc.y;

    float two_s2 = 2.0 * u_sigma * u_sigma;
    vec4  acc    = vec4(0.0);
    float wsum   = 0.0;

    for (int i = -R; i <= R; i++) {
        float w  = exp(-float(i * i) / two_s2);
        int   sx = clamp(x + i * u_step.x, 0, u_in_w - 1);
        int   sy = clamp(y + i * u_step.y, 0, u_in_h - 1);
        vec4  v  = texelFetch(u_input, ivec2(sx * u_c_groups + c_group, sy), 0);
        acc  += w * v;
        wsum += w;
    }
    fragColor = acc / wsum;
}
