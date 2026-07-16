// MULTI-FACE box decode from 5-keypoint heatmaps (f32) → 1×K×4 box tensor.
// The multi-face sibling of face_box.wgsl (which is single-face: one global
// argmax per channel + hull, so two faces merge into one box).
//
// Stage 1 (threads 0-4, one channel each): 3×3 local-maximum candidates above
//   thresh, top-MAX_CAND by peak score, refined with the windowed soft-argmax
//   centroid (the required decode — hard argmax cell-snaps and jitters the
//   crop), deduped by refined position.
// Stage 2 (thread 0): every (L-eye, R-eye) candidate pair is a face hypothesis.
//   The pair fixes the face's center, scale (interocular distance) and roll, so
//   nose + mouth corners are PREDICTED from the ArcFace canonical template and
//   matched against candidates within tol × interocular. Cross-pairing (face A's
//   left eye with face B's right eye) survives stage 1 but dies here: the
//   implied scale is wrong and there's no nose/mouth support where the template
//   says there should be.
// Stage 3/4 (thread 0): score = mean matched peak − LAMBDA · mean normalized
//   residual; greedy NMS rejects any hypothesis reusing a consumed candidate.
//
// Output slot i: (cx, cy, halfSide, score) in FRAME FRACTIONS, halfSide as a
// fraction of frame WIDTH (the grid shares the frame's aspect, so squaring in
// grid px squares in frame px). Slots past the face count are zeroed — score 0
// means "no face", which every consumer already gates on.
//
// CPU reference (kept in sync deliberately): sdk/demo/face.ts findCandidates /
// groupFaces.

struct Params {
    h         : u32,
    w         : u32,
    win       : i32,
    thresh    : f32,
    box_scale : f32,
    max_faces : u32,
    tol       : f32,
}

@group(0) @binding(0) var<storage, read>       hm_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params  : Params;
@group(0) @binding(2) var<storage, read_write> out_buf : array<vec4<f32>>;

const N_KP     : u32 = 5u;
const MAX_CAND : u32 = 5u;    // candidates per channel
const LAMBDA   : f32 = 0.5;   // geometric-residual weight
const MIN_SEP  : f32 = 1.0;   // cells — closer refined centroids are one blob
const MIN_EYE  : f32 = 1.0;   // cells — below this the grid can't resolve a pair

// Candidate table, N_KP × MAX_CAND, filled by stage 1 (thread k owns row k).
var<workgroup> c_x : array<f32, 25>;   // refined cell coords
var<workgroup> c_y : array<f32, 25>;
var<workgroup> c_s : array<f32, 25>;   // peak scores
var<workgroup> c_n : array<u32, 5>;    // candidates found per channel

fn hm_at(x: u32, y: u32, k: u32) -> f32 {
    var v = hm_buf[(y * params.w + x) * 2u + (k / 4u)];   // 8ch → 2 vec4 groups
    return v[k % 4u];
}

// ArcFace canonical 5-point template in the EYE FRAME: origin = L-eye, +u along
// the L-eye→R-eye vector, +w perpendicular (image-down for an upright face),
// both normalized by interocular distance. t: 0 = nose, 1 = L-mouth, 2 = R-mouth.
fn tmpl_u(t: u32) -> f32 {
    if (t == 0u) { return 0.50002; }
    if (t == 1u) { return 0.08598; }
    return 0.91410;
}
fn tmpl_w(t: u32) -> f32 {
    if (t == 0u) { return 0.57150; }
    return 1.15462;   // both mouth corners sit at the same depth
}

@compute @workgroup_size(5, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let k = lid.x;

    // ── stage 1: local-maximum candidates for channel k ───────────────────
    var pr : array<u32, 5>;
    var pc : array<u32, 5>;
    var ps : array<f32, 5>;
    var n  = 0u;

    for (var y = 0u; y < params.h; y++) {
        for (var x = 0u; x < params.w; x++) {
            let v = hm_at(x, y, k);
            if (v < params.thresh) { continue; }

            // 3×3 strict local max. The index tiebreak keeps exactly one member
            // of a plateau (f16 heatmaps produce exact ties).
            var is_max = true;
            for (var dy: i32 = -1; dy <= 1; dy++) {
                for (var dx: i32 = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) { continue; }
                    let rr = i32(y) + dy;
                    let cc = i32(x) + dx;
                    if (rr < 0 || rr >= i32(params.h) || cc < 0 || cc >= i32(params.w)) { continue; }
                    let u = hm_at(u32(cc), u32(rr), k);
                    if (u > v || (u == v && u32(rr) * params.w + u32(cc) < y * params.w + x)) {
                        is_max = false;
                    }
                }
            }
            if (!is_max) { continue; }

            // Insert into the score-ordered top-MAX_CAND list.
            var slot = n;
            if (n == MAX_CAND) {
                if (v <= ps[MAX_CAND - 1u]) { continue; }
                slot = MAX_CAND - 1u;
            } else {
                n = n + 1u;
            }
            while (slot > 0u && ps[slot - 1u] < v) {
                ps[slot] = ps[slot - 1u];
                pr[slot] = pr[slot - 1u];
                pc[slot] = pc[slot - 1u];
                slot = slot - 1u;
            }
            ps[slot] = v; pr[slot] = y; pc[slot] = x;
        }
    }

    // Refine each peak with the windowed soft-argmax centroid, dropping any that
    // lands on top of a higher-scoring one (two maxima on a single blob).
    var m = 0u;
    for (var i = 0u; i < n; i++) {
        var rx = f32(pc[i]);
        var ry = f32(pr[i]);
        if (params.win > 0) {
            let r0 = u32(max(0, i32(pr[i]) - params.win));
            let r1 = u32(min(i32(params.h), i32(pr[i]) + params.win + 1));
            let c0 = u32(max(0, i32(pc[i]) - params.win));
            let c1 = u32(min(i32(params.w), i32(pc[i]) + params.win + 1));
            var wsum = 0.0; var sy = 0.0; var sx = 0.0;
            for (var y = r0; y < r1; y++) {
                for (var x = c0; x < c1; x++) {
                    let v = hm_at(x, y, k);
                    wsum += v; sy += v * f32(y); sx += v * f32(x);
                }
            }
            wsum = max(wsum, 1e-6);
            rx = sx / wsum; ry = sy / wsum;
        }
        var dup = false;
        for (var j = 0u; j < m; j++) {
            let dx = c_x[k * MAX_CAND + j] - rx;
            let dy = c_y[k * MAX_CAND + j] - ry;
            if (sqrt(dx * dx + dy * dy) < MIN_SEP) { dup = true; }
        }
        if (dup) { continue; }
        c_x[k * MAX_CAND + m] = rx;
        c_y[k * MAX_CAND + m] = ry;
        c_s[k * MAX_CAND + m] = ps[i];
        m = m + 1u;
    }
    c_n[k] = m;

    workgroupBarrier();
    if (k != 0u) { return; }

    // ── stage 2: eye-pair hypotheses ──────────────────────────────────────
    var h_score : array<f32, 25>;
    var h_kp    : array<i32, 125>;   // hypothesis q, channel t → candidate idx (-1 = unmatched)
    var nh = 0u;
    let max_eye = f32(params.w) * 0.45;   // a face wider than ~half the frame

    for (var i = 0u; i < c_n[0]; i++) {
        for (var j = 0u; j < c_n[1]; j++) {
            let ax = c_x[i];
            let ay = c_y[i];
            let bx = c_x[MAX_CAND + j];
            let by = c_y[MAX_CAND + j];
            let evx = bx - ax;
            let evy = by - ay;
            let len = sqrt(evx * evx + evy * evy);
            if (len < MIN_EYE || len > max_eye) { continue; }
            let ex = vec2<f32>(evx / len, evy / len);
            let ey = vec2<f32>(-ex.y, ex.x);

            var sc      = c_s[i] + c_s[MAX_CAND + j];
            var resid   = 0.0;
            var n_match = 2u;
            var mouths  = 0u;
            var nose    = -1;
            var mo      = array<i32, 2>(-1, -1);

            for (var t = 0u; t < 3u; t++) {
                let kk = t + 2u;
                let px = ax + len * (tmpl_u(t) * ex.x + tmpl_w(t) * ey.x);
                let py = ay + len * (tmpl_u(t) * ex.y + tmpl_w(t) * ey.y);
                var best   = -1;
                var best_d = params.tol * len;
                for (var q = 0u; q < c_n[kk]; q++) {
                    let dx = c_x[kk * MAX_CAND + q] - px;
                    let dy = c_y[kk * MAX_CAND + q] - py;
                    let d  = sqrt(dx * dx + dy * dy);
                    if (d < best_d) { best_d = d; best = i32(q); }
                }
                if (best < 0) { continue; }
                sc += c_s[kk * MAX_CAND + u32(best)];
                resid += best_d / len;
                n_match += 1u;
                if (t == 0u) { nose = best; } else { mo[t - 1u] = best; mouths += 1u; }
            }
            // Support floor: a hypothesis needs the nose and at least one mouth
            // corner, otherwise two stray eye peaks become a face.
            if (nose < 0 || mouths < 1u) { continue; }

            h_score[nh] = sc / f32(n_match) - LAMBDA * (resid / f32(n_match - 2u));
            h_kp[nh * N_KP + 0u] = i32(i);
            h_kp[nh * N_KP + 1u] = i32(j);
            h_kp[nh * N_KP + 2u] = nose;
            h_kp[nh * N_KP + 3u] = mo[0];
            h_kp[nh * N_KP + 4u] = mo[1];
            nh = nh + 1u;
        }
    }

    // ── stage 3/4: greedy NMS by score → boxes ────────────────────────────
    var used : array<bool, 25>;
    for (var i = 0u; i < N_KP * MAX_CAND; i++) { used[i] = false; }
    var out_n = 0u;

    for (var iter = 0u; iter < 25u; iter++) {
        if (out_n >= params.max_faces) { break; }

        // Best remaining hypothesis. Consumed ones are marked -1e9; real scores
        // are bounded well above -1e8 (mean peak ∈ [0,1] minus LAMBDA·tol).
        var best   = -1;
        var best_s = -1e8;
        for (var q = 0u; q < nh; q++) {
            if (h_score[q] > best_s) { best_s = h_score[q]; best = i32(q); }
        }
        if (best < 0) { break; }
        let q = u32(best);
        h_score[q] = -1e9;

        var clash = false;
        for (var t = 0u; t < N_KP; t++) {
            let ci = h_kp[q * N_KP + t];
            if (ci < 0) { continue; }
            if (used[t * MAX_CAND + u32(ci)]) { clash = true; }
        }
        if (clash) { continue; }

        // Hull over this face's matched keypoints, in grid cells.
        var x0 = 1e9; var x1 = -1e9;
        var y0 = 1e9; var y1 = -1e9;
        var score = 1e9;
        for (var t = 0u; t < N_KP; t++) {
            let ci = h_kp[q * N_KP + t];
            if (ci < 0) { continue; }
            let e = t * MAX_CAND + u32(ci);
            used[e] = true;
            x0 = min(x0, c_x[e]); x1 = max(x1, c_x[e]);
            y0 = min(y0, c_y[e]); y1 = max(y1, c_y[e]);
            score = min(score, c_s[e]);
        }

        let half_px = 0.5 * params.box_scale * max(x1 - x0, y1 - y0);
        let cx = ((x0 + x1) * 0.5 + 0.5) / f32(params.w);   // +0.5: cell → center
        let cy = ((y0 + y1) * 0.5 + 0.5) / f32(params.h);
        out_buf[out_n] = vec4<f32>(cx, cy, half_px / f32(params.w), score);
        out_n = out_n + 1u;
    }

    for (var i = out_n; i < params.max_faces; i++) {
        out_buf[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
