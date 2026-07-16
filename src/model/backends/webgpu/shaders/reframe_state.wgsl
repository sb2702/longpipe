// Auto-reframe state update (f32) — one thread, one frame of camera work.
// In:  boxes (1×K×4, FaceBoxesFromHeatmaps), prev state, command.
// Out: new state = (cx, cy, size, moving) — cx/cy/size all FRAME FRACTIONS.
//
// `size` is a fraction of BOTH width and height, which is what preserves the
// frame's aspect: equal fractions of a 640×400 frame is still 8:5.
//
// The rule (tuned in demo/reframe.ts):
//   centre = frameCentre + gravity·(subject − frameCentre)   — pull, never centre
//   crop   = frame / zoom
//   constraints: the crop stays INSIDE the frame AND contains the subject+margin
// Neither is a special case; "head in the corner does nothing" falls out of them:
// once the crop centre clamps to the frame edge, containment reduces to
// x >= half+margin regardless of zoom, so an edge subject simply never fits and
// the solve falls through to its full-frame fallback (= no reframe). The zoom
// relaxation earns its keep elsewhere — backing off for a subject too BIG for
// the requested crop.
//
// Smoothing is deadband + ease: hold, move deliberately, hold. A plain EMA reads
// as swimming as the subject breathes.
//
// State is threaded across frames by the renderer with copyTensor — same carrier
// pattern as the flow stabilizer's stabPrev. No readback anywhere.

struct Params {
    k        : u32,
    zoom     : f32,
    gravity  : f32,
    margin   : f32,
    deadband : f32,
    ease     : f32,
    aspect   : f32,   // frame w/h — the box's halfSide is a fraction of WIDTH
    _pad     : f32,
}

// cmd.x — 0 auto (track), 1 manual-hold (frozen), 2 manual-solve (snap once).
const MODE_HOLD  : f32 = 1.0;
const MODE_SOLVE : f32 = 2.0;

@group(0) @binding(0) var<storage, read>       box_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       prev_buf : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       cmd_buf  : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params   : Params;
@group(0) @binding(4) var<storage, read_write> out_buf  : array<vec4<f32>>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let prev = prev_buf[0];
    let mode = cmd_buf[0].x;

    // Subject = largest live face. NOTE: no hysteresis — two faces of near-equal
    // size can hand off between frames. Add a stickiness term here if it bites.
    var best = -1;
    var best_hs = 0.0;
    for (var i = 0u; i < params.k; i++) {
        let b = box_buf[i];
        if (b.w > 0.0 && b.z > best_hs) { best_hs = b.z; best = i32(i); }
    }
    if (best < 0) { out_buf[0] = prev; return; }   // no subject → hold
    let subj = box_buf[u32(best)];

    // Solve. Stepped relaxation rather than closed-form: the frame-edge clamp
    // makes the centre depend on the crop size, so it's implicit.
    let half_x = subj.z;
    let half_y = subj.z * params.aspect;
    // Fallback = the FULL frame, i.e. no reframe at all. A subject too close to
    // the edge to contain can't be rescued by relaxing zoom (once the crop centre
    // clamps to the frame edge, the containment test reduces to x >= half+margin,
    // independent of zoom) — so it falls through to this and eases out to the raw
    // frame. That IS "head in the corner does nothing"; holding the last crop
    // instead would freeze on a shot the subject is walking out of.
    //
    // What the relaxation actually buys is the other axis: a subject too BIG for
    // the requested zoom (a close-up) backs the crop off until they fit.
    var w_cx = 0.5; var w_cy = 0.5; var w_s = 1.0;
    for (var t = 0; t <= 25; t++) {
        let z = params.zoom + (1.0 - params.zoom) * (f32(t) / 25.0);
        let s = 1.0 / z;
        let cx = clamp(0.5 + params.gravity * (subj.x - 0.5), s * 0.5, 1.0 - s * 0.5);
        let cy = clamp(0.5 + params.gravity * (subj.y - 0.5), s * 0.5, 1.0 - s * 0.5);
        if (abs(subj.x - cx) + half_x + params.margin <= s * 0.5 &&
            abs(subj.y - cy) + half_y + params.margin <= s * 0.5) {
            w_cx = cx; w_cy = cy; w_s = s;
            break;
        }
    }
    // size 0 = uninitialised: snap rather than easing in from nothing. This is
    // also what makes manual mode "reframe once when enabled, then freeze".
    if (prev.z <= 0.0 || mode == MODE_SOLVE) {
        out_buf[0] = vec4<f32>(w_cx, w_cy, w_s, 0.0);
        return;
    }
    if (mode == MODE_HOLD) { out_buf[0] = prev; return; }

    let d = length(vec2<f32>(w_cx - prev.x, w_cy - prev.y)) + abs(w_s - prev.z);
    var moving = prev.w;
    if (d > params.deadband) { moving = 1.0; }
    else if (d < params.deadband * 0.3) { moving = 0.0; }
    if (moving < 0.5) { out_buf[0] = vec4<f32>(prev.xyz, 0.0); return; }

    out_buf[0] = vec4<f32>(
        mix(prev.x, w_cx, params.ease),
        mix(prev.y, w_cy, params.ease),
        mix(prev.z, w_s,  params.ease),
        1.0,
    );
}
