#version 300 es
// MULTI-FACE box decode from 5-keypoint heatmaps → 1×K texture, slot i =
// (cx, cy, halfSide, score) in FRAME FRACTIONS (halfSide a fraction of frame
// WIDTH). The multi-face sibling of face_box.glsl. See face_boxes.wgsl for the
// full algorithm commentary — this mirrors it exactly.
//
// WebGL has no compute and no shared memory, so each fragment (= one face slot)
// independently redoes the ENTIRE decode — candidates for all 5 channels,
// hypotheses, NMS — and emits the slot matching its own x. That's K× redundant
// work, which is fine here: the grid is ≤ ~48×28 and K ≤ 6, and it keeps the
// WGSL and GLSL paths line-for-line comparable.
//
// Heatmap texture: (w * 2, h) — 8 channels = 2 vec4 groups per pixel.

precision highp float;
precision highp int;

uniform sampler2D u_hm;
uniform int u_h;
uniform int u_w;
uniform int u_win;
uniform float u_thresh;
uniform float u_box_scale;
uniform int u_max_faces;
uniform float u_tol;

out vec4 fragColor;

const int N_KP     = 5;
const int MAX_CAND = 5;
const float LAMBDA  = 0.5;
const float MIN_SEP = 1.0;
const float MIN_EYE = 1.0;

float c_x[25];
float c_y[25];
float c_s[25];
int   c_n[5];

float hmAt(int x, int y, int k) {
    vec4 v = texelFetch(u_hm, ivec2(x * 2 + k / 4, y), 0);
    return v[k % 4];
}

// ArcFace canonical template in the eye frame (origin = L-eye, +u along the
// L→R eye vector, +w perpendicular), normalized by interocular distance.
float tmplU(int t) { return t == 0 ? 0.50002 : (t == 1 ? 0.08598 : 0.91410); }
float tmplW(int t) { return t == 0 ? 0.57150 : 1.15462; }

void main() {
    int myFace = int(gl_FragCoord.x);

    // ── stage 1: local-maximum candidates, per channel ────────────────────
    for (int k = 0; k < N_KP; k++) {
        int   pr[5];
        int   pc[5];
        float ps[5];
        int n = 0;

        for (int y = 0; y < u_h; y++) {
            for (int x = 0; x < u_w; x++) {
                float v = hmAt(x, y, k);
                if (v < u_thresh) continue;

                bool isMax = true;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int rr = y + dy, cc = x + dx;
                        if (rr < 0 || rr >= u_h || cc < 0 || cc >= u_w) continue;
                        float u = hmAt(cc, rr, k);
                        if (u > v || (u == v && rr * u_w + cc < y * u_w + x)) isMax = false;
                    }
                }
                if (!isMax) continue;

                int slot = n;
                if (n == MAX_CAND) {
                    if (v <= ps[MAX_CAND - 1]) continue;
                    slot = MAX_CAND - 1;
                } else {
                    n++;
                }
                while (slot > 0 && ps[slot - 1] < v) {
                    ps[slot] = ps[slot - 1]; pr[slot] = pr[slot - 1]; pc[slot] = pc[slot - 1];
                    slot--;
                }
                ps[slot] = v; pr[slot] = y; pc[slot] = x;
            }
        }

        int m = 0;
        for (int i = 0; i < n; i++) {
            float rx = float(pc[i]);
            float ry = float(pr[i]);
            if (u_win > 0) {
                int r0 = max(0, pr[i] - u_win), r1 = min(u_h, pr[i] + u_win + 1);
                int c0 = max(0, pc[i] - u_win), c1 = min(u_w, pc[i] + u_win + 1);
                float wsum = 0.0, sy = 0.0, sx = 0.0;
                for (int y = r0; y < r1; y++) {
                    for (int x = c0; x < c1; x++) {
                        float v = hmAt(x, y, k);
                        wsum += v; sy += v * float(y); sx += v * float(x);
                    }
                }
                wsum = max(wsum, 1e-6);
                rx = sx / wsum; ry = sy / wsum;
            }
            bool dup = false;
            for (int j = 0; j < m; j++) {
                float dx = c_x[k * MAX_CAND + j] - rx;
                float dy = c_y[k * MAX_CAND + j] - ry;
                if (sqrt(dx * dx + dy * dy) < MIN_SEP) dup = true;
            }
            if (dup) continue;
            c_x[k * MAX_CAND + m] = rx;
            c_y[k * MAX_CAND + m] = ry;
            c_s[k * MAX_CAND + m] = ps[i];
            m++;
        }
        c_n[k] = m;
    }

    // ── stage 2: eye-pair hypotheses ──────────────────────────────────────
    float h_score[25];
    int   h_kp[125];
    int nh = 0;
    float maxEye = float(u_w) * 0.45;

    for (int i = 0; i < c_n[0]; i++) {
        for (int j = 0; j < c_n[1]; j++) {
            float ax = c_x[i], ay = c_y[i];
            float bx = c_x[MAX_CAND + j], by = c_y[MAX_CAND + j];
            float evx = bx - ax, evy = by - ay;
            float len = sqrt(evx * evx + evy * evy);
            if (len < MIN_EYE || len > maxEye) continue;
            vec2 ex = vec2(evx / len, evy / len);
            vec2 ey = vec2(-ex.y, ex.x);

            float sc = c_s[i] + c_s[MAX_CAND + j];
            float resid = 0.0;
            int nMatch = 2, mouths = 0, nose = -1;
            int mo[2];
            mo[0] = -1; mo[1] = -1;

            for (int t = 0; t < 3; t++) {
                int kk = t + 2;
                float px = ax + len * (tmplU(t) * ex.x + tmplW(t) * ey.x);
                float py = ay + len * (tmplU(t) * ex.y + tmplW(t) * ey.y);
                int best = -1;
                float bestD = u_tol * len;
                for (int q = 0; q < c_n[kk]; q++) {
                    float dx = c_x[kk * MAX_CAND + q] - px;
                    float dy = c_y[kk * MAX_CAND + q] - py;
                    float d = sqrt(dx * dx + dy * dy);
                    if (d < bestD) { bestD = d; best = q; }
                }
                if (best < 0) continue;
                sc += c_s[kk * MAX_CAND + best];
                resid += bestD / len;
                nMatch++;
                if (t == 0) nose = best; else { mo[t - 1] = best; mouths++; }
            }
            if (nose < 0 || mouths < 1) continue;

            h_score[nh] = sc / float(nMatch) - LAMBDA * (resid / float(nMatch - 2));
            h_kp[nh * N_KP + 0] = i;
            h_kp[nh * N_KP + 1] = j;
            h_kp[nh * N_KP + 2] = nose;
            h_kp[nh * N_KP + 3] = mo[0];
            h_kp[nh * N_KP + 4] = mo[1];
            nh++;
        }
    }

    // ── stage 3/4: greedy NMS → the box for slot myFace ───────────────────
    bool used[25];
    for (int i = 0; i < N_KP * MAX_CAND; i++) used[i] = false;
    int outN = 0;

    for (int iter = 0; iter < 25; iter++) {
        if (outN >= u_max_faces) break;

        int best = -1;
        float bestS = -1e8;
        for (int q = 0; q < nh; q++) {
            if (h_score[q] > bestS) { bestS = h_score[q]; best = q; }
        }
        if (best < 0) break;
        h_score[best] = -1e9;

        bool clash = false;
        for (int t = 0; t < N_KP; t++) {
            int ci = h_kp[best * N_KP + t];
            if (ci < 0) continue;
            if (used[t * MAX_CAND + ci]) clash = true;
        }
        if (clash) continue;

        float x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, score = 1e9;
        for (int t = 0; t < N_KP; t++) {
            int ci = h_kp[best * N_KP + t];
            if (ci < 0) continue;
            int e = t * MAX_CAND + ci;
            used[e] = true;
            x0 = min(x0, c_x[e]); x1 = max(x1, c_x[e]);
            y0 = min(y0, c_y[e]); y1 = max(y1, c_y[e]);
            score = min(score, c_s[e]);
        }

        if (outN == myFace) {
            float halfPx = 0.5 * u_box_scale * max(x1 - x0, y1 - y0);
            float cx = ((x0 + x1) * 0.5 + 0.5) / float(u_w);
            float cy = ((y0 + y1) * 0.5 + 0.5) / float(u_h);
            fragColor = vec4(cx, cy, halfPx / float(u_w), score);
            return;
        }
        outN++;
    }

    fragColor = vec4(0.0);   // no face in this slot
}
