enable f16;

// Box-driven square crop + bilinear resample + channel normalization (f16 storage, f32 math).
// Reads the box tensor (FaceBoxFromHeatmaps output: cx, cy, halfSide, score in
// frame fractions; halfSide as a fraction of frame WIDTH) and emits the
// landmark model's input: ((rgb - mean) / std), .w = 0.
//
// Sampling mirrors landmark training's warpAffine: src = center + (u - out/2)
// · side/out — u is the integer output pixel index (cv2 convention).

struct Params {
    in_h  : u32,
    in_w  : u32,
    out_h : u32,
    out_w : u32,
    slot  : u32,        // box-tensor slot (multi-face); 0 for the single-face path
    pad0  : u32,
    pad1  : u32,
    pad2  : u32,
    mean  : vec4<f32>,   // .xyz used
    stdev : vec4<f32>,   // .xyz used
}

@group(0) @binding(0) var<storage, read>       frame_buf : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       box_buf   : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params    : Params;
@group(0) @binding(3) var<storage, read_write> out_buf   : array<vec4<f16>>;

fn samp(x: i32, y: i32, W: i32, H: i32) -> vec4<f32> {
    let cx = clamp(x, 0, W - 1);
    let cy = clamp(y, 0, H - 1);
    return vec4<f32>(frame_buf[u32(cy) * u32(W) + u32(cx)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.out_w || y >= params.out_h) { return; }

    let W = i32(params.in_w);
    let H = i32(params.in_h);
    let box = vec4<f32>(box_buf[params.slot]);
    let cx   = box.x * f32(params.in_w);
    let cy   = box.y * f32(params.in_h);
    let side = 2.0 * box.z * f32(params.in_w);   // square in frame px

    let sx = clamp(cx + (f32(x) - f32(params.out_w) * 0.5) * side / f32(params.out_w), 0.0, f32(W - 1));
    let sy = clamp(cy + (f32(y) - f32(params.out_h) * 0.5) * side / f32(params.out_h), 0.0, f32(H - 1));

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);

    let top = mix(samp(x0, y0, W, H), samp(x0 + 1, y0, W, H), tx);
    let bot = mix(samp(x0, y0 + 1, W, H), samp(x0 + 1, y0 + 1, W, H), tx);
    let v = mix(top, bot, ty);

    let n = (v.rgb - params.mean.rgb) / params.stdev.rgb;
    out_buf[y * params.out_w + x] = vec4<f16>(vec4<f32>(n, 0.0));
}
