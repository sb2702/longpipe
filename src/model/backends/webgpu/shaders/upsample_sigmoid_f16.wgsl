enable f16;

// Bilinear upsample + sigmoid fused — full f16 variant.
// Interpolation computed in f32 for accuracy; sigmoid and result cast to f16.

struct Params {
    in_h           : u32,
    in_w           : u32,
    out_h          : u32,
    out_w          : u32,
    channel_groups : u32,
    _pad0          : u32,
    _pad1          : u32,
    _pad2          : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output_buf : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params     : Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let c  = gid.z;

    if (ox >= params.out_w || oy >= params.out_h || c >= params.channel_groups) {
        return;
    }

    let IH = params.in_h;
    let IW = params.in_w;
    let C  = params.channel_groups;

    let src_x = (f32(ox) + 0.5) * (f32(IW) / f32(params.out_w)) - 0.5;
    let src_y = (f32(oy) + 0.5) * (f32(IH) / f32(params.out_h)) - 0.5;

    let x0 = u32(clamp(i32(floor(src_x)),     0, i32(IW) - 1));
    let x1 = u32(clamp(i32(floor(src_x)) + 1, 0, i32(IW) - 1));
    let y0 = u32(clamp(i32(floor(src_y)),     0, i32(IH) - 1));
    let y1 = u32(clamp(i32(floor(src_y)) + 1, 0, i32(IH) - 1));

    let wx = src_x - floor(src_x);
    let wy = src_y - floor(src_y);

    let tl = vec4<f32>(input_buf[y0 * IW * C + x0 * C + c]);
    let tr = vec4<f32>(input_buf[y0 * IW * C + x1 * C + c]);
    let bl = vec4<f32>(input_buf[y1 * IW * C + x0 * C + c]);
    let br = vec4<f32>(input_buf[y1 * IW * C + x1 * C + c]);

    let result = (1.0 - wy) * ((1.0 - wx) * tl + wx * tr)
               +        wy  * ((1.0 - wx) * bl + wx * br);

    output_buf[oy * params.out_w * C + ox * C + c] = vec4<f16>(1.0 / (1.0 + exp(-result)));
}
