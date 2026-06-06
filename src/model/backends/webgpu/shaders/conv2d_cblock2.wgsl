// Conv2d, output-channel-blocked (K=2) — drop-in replacement for conv2d.wgsl.
//
// Identical math, layout, and output to conv2d.wgsl, but each thread computes 2
// output channel-groups (8 channels) for one pixel, loading each input vec4 ONCE
// and reusing it across both groups' mat4x4 weight blocks in registers. Half the
// register pressure of the K=4 variant (2 accumulators vs 4). Output is bit-
// identical to conv2d.wgsl (≤1 ULP from FMA contraction). Dispatch
// ceil(out_groups / 2) in z (see conv2d.ts).
//
// Tensor layout: NHWC, channels in vec4 groups (same as conv2d.wgsl).
// Weight layout: [K*K][out_groups][in_groups] array of mat4x4 (same as conv2d.wgsl).

struct Params {
    in_h        : u32,
    in_w        : u32,
    out_h       : u32,
    out_w       : u32,
    in_groups   : u32,   // in_channels / 4
    out_groups  : u32,   // out_channels / 4
    kernel_h    : u32,
    kernel_w    : u32,
    stride      : u32,
    pad_top     : u32,
    pad_left    : u32,
    activation  : u32,   // 0 = none, 1 = relu6, 2 = relu, 3 = leaky
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       weight_buf : array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read>       bias_buf   : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params     : Params;
@group(0) @binding(4) var<storage, read_write> output_buf : array<vec4<f32>>;

const KB = 2u;

fn act(v: vec4<f32>, a: u32) -> vec4<f32> {
    if (a == 1u) { return clamp(v, vec4<f32>(0.0), vec4<f32>(6.0)); }
    if (a == 2u) { return max(v, vec4<f32>(0.0)); }
    if (a == 3u) { return max(v, 0.1 * v); }   // leaky relu (slope 0.1)
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x  = gid.x;       // output column
    let y  = gid.y;       // output row
    let o0 = gid.z * KB;  // first output channel group

    if (x >= params.out_w || y >= params.out_h || o0 >= params.out_groups) {
        return;
    }

    let I = params.in_groups;
    let O = params.out_groups;
    // Tail handling when out_groups is odd: clamp the spare lane to o0 (valid
    // index, no OOB) and just don't write it.
    let has1 = (o0 + 1u) < O;
    let o1 = select(o0, o0 + 1u, has1);

    var acc0 = bias_buf[o0];
    var acc1 = bias_buf[o1];

    for (var ky = 0u; ky < params.kernel_h; ky++) {
        for (var kx = 0u; kx < params.kernel_w; kx++) {
            let in_y_s = i32(y * params.stride + ky) - i32(params.pad_top);
            let in_x_s = i32(x * params.stride + kx) - i32(params.pad_left);

            if (in_y_s < 0 || in_x_s < 0 ||
                u32(in_y_s) >= params.in_h || u32(in_x_s) >= params.in_w) {
                continue;
            }

            let z   = ky * params.kernel_w + kx;
            let inB = u32(in_y_s) * params.in_w * I + u32(in_x_s) * I;
            let wb0 = z * O * I + o0 * I;
            let wb1 = z * O * I + o1 * I;
            for (var i = 0u; i < I; i++) {
                let iv = input_buf[inB + i];   // loaded ONCE, reused across 2 groups
                acc0 += weight_buf[wb0 + i] * iv;
                acc1 += weight_buf[wb1 + i] * iv;
            }
        }
    }

    let baseO = y * params.out_w * O + x * O;
    output_buf[baseO + o0] = act(acc0, params.activation);
    if (has1) { output_buf[baseO + o1] = act(acc1, params.activation); }
}
