// Bilinear upsample + 1×1 pointwise conv fused.
// For each output pixel, bilinearly samples the small input for each in_group,
// immediately applies the 1×1 conv weights, and writes the activated result.
// Eliminates the intermediate full-resolution upsample buffer.

struct Params {
    in_h       : u32,
    in_w       : u32,
    out_h      : u32,
    out_w      : u32,
    in_groups  : u32,
    out_groups : u32,
    activation : u32,
    _pad0      : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weight_buf : array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read>       bias_buf   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params     : Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let og = gid.z;

    if (ox >= params.out_w || oy >= params.out_h || og >= params.out_groups) { return; }

    let IH = params.in_h;
    let IW = params.in_w;
    let IG = params.in_groups;

    let src_x = (f32(ox) + 0.5) * (f32(IW) / f32(params.out_w)) - 0.5;
    let src_y = (f32(oy) + 0.5) * (f32(IH) / f32(params.out_h)) - 0.5;

    let x0 = u32(clamp(i32(floor(src_x)),     0, i32(IW) - 1));
    let x1 = u32(clamp(i32(floor(src_x)) + 1, 0, i32(IW) - 1));
    let y0 = u32(clamp(i32(floor(src_y)),     0, i32(IH) - 1));
    let y1 = u32(clamp(i32(floor(src_y)) + 1, 0, i32(IH) - 1));

    let wx = src_x - floor(src_x);
    let wy = src_y - floor(src_y);

    var result = bias_buf[og];

    for (var ig = 0u; ig < IG; ig++) {
        let tl = input_buf[y0 * IW * IG + x0 * IG + ig];
        let tr = input_buf[y0 * IW * IG + x1 * IG + ig];
        let bl = input_buf[y1 * IW * IG + x0 * IG + ig];
        let br = input_buf[y1 * IW * IG + x1 * IG + ig];

        let sampled = (1.0 - wy) * ((1.0 - wx) * tl + wx * tr)
                    +        wy  * ((1.0 - wx) * bl + wx * br);

        result += weight_buf[og * IG + ig] * sampled;
    }

    if (params.activation == 1u) {
        result = clamp(result, vec4<f32>(0.0), vec4<f32>(6.0));
    } else if (params.activation == 2u) {
        result = result * clamp(result + 3.0, vec4<f32>(0.0), vec4<f32>(6.0)) / 6.0;
    }

    output_buf[oy * params.out_w * params.out_groups + ox * params.out_groups + og] = result;
}
