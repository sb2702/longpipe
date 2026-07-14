#version 300 es
// Face-box decode from 5-keypoint heatmaps. Output is a 1×1 texture:
// (cx, cy, halfSide, score) in FRAME FRACTIONS (halfSide as a fraction of
// frame WIDTH; the heatmap grid shares the frame's aspect, so squaring in
// grid px squares in frame px). Peaks refined with a windowed soft-argmax
// centroid — NEVER hard argmax (whole-cell snapping jitters the crop).
//
// Single fragment does all 5 channels serially — the grid is tiny (≤ ~48×28).
// Heatmap texture: (w * 2, h) — 8 channels = 2 vec4 groups per pixel.

precision highp float;
precision highp int;

uniform sampler2D u_hm;
uniform int u_h;
uniform int u_w;
uniform int u_win;
uniform float u_thresh;
uniform float u_box_scale;

out vec4 fragColor;

float hmAt(int x, int y, int k) {
    vec4 v = texelFetch(u_hm, ivec2(x * 2 + k / 4, y), 0);
    return v[k % 4];
}

void main() {
    float x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, score = 1e9;

    for (int k = 0; k < 5; k++) {
        // Peak.
        float peak = -1.0;
        int pr = 0, pc = 0;
        for (int y = 0; y < u_h; y++) {
            for (int x = 0; x < u_w; x++) {
                float v = hmAt(x, y, k);
                if (v > peak) { peak = v; pr = y; pc = x; }
            }
        }
        // Windowed soft-argmax centroid.
        int r0 = max(0, pr - u_win), r1 = min(u_h, pr + u_win + 1);
        int c0 = max(0, pc - u_win), c1 = min(u_w, pc + u_win + 1);
        float wsum = 0.0, sy = 0.0, sx = 0.0;
        for (int y = r0; y < r1; y++) {
            for (int x = c0; x < c1; x++) {
                float v = hmAt(x, y, k);
                wsum += v; sy += v * float(y); sx += v * float(x);
            }
        }
        wsum = max(wsum, 1e-6);
        float kx = sx / wsum, ky = sy / wsum;
        x0 = min(x0, kx); x1 = max(x1, kx);
        y0 = min(y0, ky); y1 = max(y1, ky);
        score = min(score, peak);
    }
    if (score < u_thresh) score = 0.0;

    float halfPx = 0.5 * u_box_scale * max(x1 - x0, y1 - y0);
    float cx = ((x0 + x1) * 0.5 + 0.5) / float(u_w);
    float cy = ((y0 + y1) * 0.5 + 0.5) / float(u_h);
    fragColor = vec4(cx, cy, halfPx / float(u_w), score);
}
