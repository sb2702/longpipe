#version 300 es
// Global average pool, pass 1 of 2: mean over x for each (row, channel group).
// Fragment shaders can't share work across pixels, so the pool is split into a
// row pass (out: (C_groups, H)) and a column pass (pool_cols.glsl).
precision highp float;
precision highp int;

uniform sampler2D u_input;   // (in_W * C_groups, in_H)
uniform int u_in_w;
uniform int u_c_groups;

out vec4 fragColor;

void main() {
    int c = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    vec4 sum = vec4(0.0);
    for (int x = 0; x < u_in_w; x++) sum += texelFetch(u_input, ivec2(x * u_c_groups + c, y), 0);
    fragColor = sum / float(u_in_w);
}
