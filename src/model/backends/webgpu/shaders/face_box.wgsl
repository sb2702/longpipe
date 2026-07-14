// Face-box decode from 5-keypoint heatmaps (f32). One workgroup: threads 0-4
// each soft-argmax one channel (peak + windowed centroid — hard argmax snaps
// to whole cells and jitters the crop); thread 0 combines the keypoint hull
// into a pixel-square box.
//
// Output (1 vec4): (cx, cy, halfSide, score) in FRAME FRACTIONS — halfSide as
// a fraction of frame WIDTH. The heatmap grid shares the frame's aspect, so
// squaring in grid-cell px squares in frame px. score = min keypoint peak,
// forced to 0 if any peak < thresh.

struct Params {
    h         : u32,
    w         : u32,
    win       : i32,
    thresh    : f32,
    box_scale : f32,
}

@group(0) @binding(0) var<storage, read>       hm_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params  : Params;
@group(0) @binding(2) var<storage, read_write> out_buf : array<vec4<f32>>;

var<workgroup> kp_x : array<f32, 5>;   // continuous cell coords
var<workgroup> kp_y : array<f32, 5>;
var<workgroup> kp_s : array<f32, 5>;   // peak scores

fn hm_at(x: u32, y: u32, k: u32) -> f32 {
    var v = hm_buf[(y * params.w + x) * 2u + (k / 4u)];   // 8ch → 2 vec4 groups
    return v[k % 4u];
}

@compute @workgroup_size(5, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let k = lid.x;

    // Peak.
    var peak = -1.0;
    var pr = 0u;
    var pc = 0u;
    for (var y = 0u; y < params.h; y++) {
        for (var x = 0u; x < params.w; x++) {
            let v = hm_at(x, y, k);
            if (v > peak) { peak = v; pr = y; pc = x; }
        }
    }

    // Windowed soft-argmax centroid around the peak.
    let r0 = u32(max(0, i32(pr) - params.win));
    let r1 = u32(min(i32(params.h), i32(pr) + params.win + 1));
    let c0 = u32(max(0, i32(pc) - params.win));
    let c1 = u32(min(i32(params.w), i32(pc) + params.win + 1));
    var wsum = 0.0;
    var sy = 0.0;
    var sx = 0.0;
    for (var y = r0; y < r1; y++) {
        for (var x = c0; x < c1; x++) {
            let v = hm_at(x, y, k);
            wsum += v;
            sy += v * f32(y);
            sx += v * f32(x);
        }
    }
    wsum = max(wsum, 1e-6);
    kp_x[k] = sx / wsum;
    kp_y[k] = sy / wsum;
    kp_s[k] = peak;
    workgroupBarrier();

    if (k != 0u) { return; }

    // Hull over the 5 keypoints, in grid-cell px.
    var x0 = 1e9; var x1 = -1e9;
    var y0 = 1e9; var y1 = -1e9;
    var score = 1e9;
    for (var i = 0u; i < 5u; i++) {
        x0 = min(x0, kp_x[i]); x1 = max(x1, kp_x[i]);
        y0 = min(y0, kp_y[i]); y1 = max(y1, kp_y[i]);
        score = min(score, kp_s[i]);
    }
    if (score < params.thresh) { score = 0.0; }

    // Square box: side = boxScale × hull long side (grid px == frame-px aspect).
    let half_px = 0.5 * params.box_scale * max(x1 - x0, y1 - y0);
    let cx = ((x0 + x1) * 0.5 + 0.5) / f32(params.w);   // +0.5: cell → center
    let cy = ((y0 + y1) * 0.5 + 0.5) / f32(params.h);
    out_buf[0] = vec4<f32>(cx, cy, half_px / f32(params.w), score);
}
