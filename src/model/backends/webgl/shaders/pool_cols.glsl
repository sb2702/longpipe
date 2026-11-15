#version 300 es
// Global average pool, pass 2 of 2: mean of the row means → (C_groups, 1).
precision highp float;
precision highp int;

uniform sampler2D u_input;   // (C_groups, in_H) — pool_rows output
uniform int u_in_h;

out vec4 fragColor;

void main() {
    int c = int(gl_FragCoord.x);
    vec4 sum = vec4(0.0);
    for (int y = 0; y < u_in_h; y++) sum += texelFetch(u_input, ivec2(c, y), 0);
    fragColor = sum / float(u_in_h);
}
