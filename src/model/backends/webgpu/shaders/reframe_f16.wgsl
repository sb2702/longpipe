enable f16;

// Apply the reframe crop (f16 storage, f32 math): sample `src` through the view rect. Same shape
// in and out — this is the geometric slot of the one-compositor design.
//
// Applied AFTER the effect chain (touch-up draws its mesh at FULL-frame landmark
// coords; reframing upstream would land every face effect in the wrong place),
// and to the foreground + alpha only. The background is bound separately by the
// compositor and deliberately does NOT move — you zoom toward the person, the
// virtual backdrop behind them stays put. Blur backgrounds are derived from the
// foreground, so they follow for free; no per-mode special case.
//
// rect = (cx, cy, size, moving) in frame fractions; size is a fraction of BOTH
// dims (that's what preserves aspect). size ≤ 0 means uninitialised → identity.

struct Params {
    h : u32,
    w : u32,
}

@group(0) @binding(0) var<storage, read>       src_buf  : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       rect_buf : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params   : Params;
@group(0) @binding(3) var<storage, read_write> out_buf  : array<vec4<f16>>;

fn samp(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.w) - 1);
    let cy = clamp(y, 0, i32(params.h) - 1);
    return vec4<f32>(src_buf[u32(cy) * params.w + u32(cx)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.w || gid.y >= params.h) { return; }

    let r = vec4<f32>(rect_buf[0]);
    var s = r.z; var rcx = r.x; var rcy = r.y;
    if (s <= 0.0) { s = 1.0; rcx = 0.5; rcy = 0.5; }   // identity until the first solve

    let fx = (rcx - s * 0.5) + ((f32(gid.x) + 0.5) / f32(params.w)) * s;
    let fy = (rcy - s * 0.5) + ((f32(gid.y) + 0.5) / f32(params.h)) * s;
    // frame fraction → texel index (a texel's centre sits at index + 0.5), so an
    // identity rect samples each texel exactly and the op is a bit-exact copy.
    let sx = fx * f32(params.w) - 0.5;
    let sy = fy * f32(params.h) - 0.5;

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);

    let top = mix(samp(x0, y0),     samp(x0 + 1, y0),     tx);
    let bot = mix(samp(x0, y0 + 1), samp(x0 + 1, y0 + 1), tx);
    out_buf[gid.y * params.w + gid.x] = vec4<f16>(mix(top, bot, ty));
}
