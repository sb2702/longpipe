#version 300 es
// down_adapter: stride-N 3×3 conv (mat4x4, 4→4) + relu, then 1×1 adapter
// (4→3, last row 0). down_w = 9 mat4x4 (1-row, 36 texels); adapt_w = 1 mat4x4
// (4 texels). Output 3ch in .xyz (.w=0). Symmetric pad. Weight packing matches
// conv2d (col-major mat4x4 via the flat accessor).

precision highp float;
precision highp int;

uniform sampler2D u_input;     // (in_W, in_H) — in_c padded to 1 vec4
uniform sampler2D u_down_w;    // (36, 1)
uniform sampler2D u_down_b;    // (1, 1)
uniform sampler2D u_adapt_w;   // (4, 1)
uniform sampler2D u_adapt_b;   // (1, 1) — .xyz
uniform int u_in_w;
uniform int u_in_h;
uniform int u_stride;
uniform int u_pad;

out vec4 fragColor;

float dwf(int i) { return texelFetch(u_down_w,  ivec2(i / 4, 0), 0)[i & 3]; }
float awf(int i) { return texelFetch(u_adapt_w, ivec2(i / 4, 0), 0)[i & 3]; }

void main() {
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);

    vec4 d = texelFetch(u_down_b, ivec2(0, 0), 0);
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int iy = y * u_stride + ky - u_pad;
            int ix = x * u_stride + kx - u_pad;
            if (iy < 0 || ix < 0 || iy >= u_in_h || ix >= u_in_w) continue;
            int kpos = ky * 3 + kx;
            vec4 v = texelFetch(u_input, ivec2(ix, iy), 0);
            int m = kpos * 16;
            d.x += dwf(m + 0) * v.x + dwf(m + 4) * v.y + dwf(m + 8)  * v.z + dwf(m + 12) * v.w;
            d.y += dwf(m + 1) * v.x + dwf(m + 5) * v.y + dwf(m + 9)  * v.z + dwf(m + 13) * v.w;
            d.z += dwf(m + 2) * v.x + dwf(m + 6) * v.y + dwf(m + 10) * v.z + dwf(m + 14) * v.w;
            d.w += dwf(m + 3) * v.x + dwf(m + 7) * v.y + dwf(m + 11) * v.z + dwf(m + 15) * v.w;
        }
    }
    d = max(d, vec4(0.0));   // relu

    vec4 ab = texelFetch(u_adapt_b, ivec2(0, 0), 0);
    vec4 a;
    a.x = awf(0) * d.x + awf(4) * d.y + awf(8)  * d.z + awf(12) * d.w + ab.x;
    a.y = awf(1) * d.x + awf(5) * d.y + awf(9)  * d.z + awf(13) * d.w + ab.y;
    a.z = awf(2) * d.x + awf(6) * d.y + awf(10) * d.z + awf(14) * d.w + ab.z;
    fragColor = vec4(a.xyz, 0.0);
}
