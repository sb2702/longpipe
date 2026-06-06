enable f16;

// Conv2d, output-channel-blocked (K=4) — full f16. Drop-in replacement for
// conv2d_f16.wgsl. Each thread computes 4 output channel-groups for one pixel,
// loading each input vec4 once and reusing it across all four weight blocks.
// Output is bit-identical to conv2d_f16.wgsl. Dispatch ceil(out_groups / 4) in z.

struct Params {
    in_h        : u32,
    in_w        : u32,
    out_h       : u32,
    out_w       : u32,
    in_groups   : u32,
    out_groups  : u32,
    kernel_h    : u32,
    kernel_w    : u32,
    stride      : u32,
    pad_top     : u32,
    pad_left    : u32,
    activation  : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       weight_buf : array<mat4x4<f16>>;
@group(0) @binding(2) var<storage, read>       bias_buf   : array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             params     : Params;
@group(0) @binding(4) var<storage, read_write> output_buf : array<vec4<f16>>;

const KB = 4u;

// Accumulate in f32 (storage stays f16) — the BN-free flow net sums ~1000+ terms
// with cancellation, which f16 accumulation can't hold. Convs are memory-bound so
// this is ~free. act runs in f32, then narrows to f16 for output.
fn act(v: vec4<f32>, a: u32) -> vec4<f16> {
    if (a == 1u) { return vec4<f16>(clamp(v, vec4<f32>(0.0), vec4<f32>(6.0))); }
    if (a == 2u) { return vec4<f16>(max(v, vec4<f32>(0.0))); }
    if (a == 3u) { return vec4<f16>(max(v, 0.1 * v)); }   // leaky relu (slope 0.1)
    return vec4<f16>(v);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x  = gid.x;
    let y  = gid.y;
    let o0 = gid.z * KB;

    if (x >= params.out_w || y >= params.out_h || o0 >= params.out_groups) {
        return;
    }

    let I = params.in_groups;
    let O = params.out_groups;
    let has1 = (o0 + 1u) < O;
    let has2 = (o0 + 2u) < O;
    let has3 = (o0 + 3u) < O;
    let o1 = select(o0, o0 + 1u, has1);
    let o2 = select(o0, o0 + 2u, has2);
    let o3 = select(o0, o0 + 3u, has3);

    var acc0 = vec4<f32>(bias_buf[o0]);
    var acc1 = vec4<f32>(bias_buf[o1]);
    var acc2 = vec4<f32>(bias_buf[o2]);
    var acc3 = vec4<f32>(bias_buf[o3]);

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
            let wb2 = z * O * I + o2 * I;
            let wb3 = z * O * I + o3 * I;
            for (var i = 0u; i < I; i++) {
                let iv = input_buf[inB + i];
                acc0 += vec4<f32>(weight_buf[wb0 + i] * iv);
                acc1 += vec4<f32>(weight_buf[wb1 + i] * iv);
                acc2 += vec4<f32>(weight_buf[wb2 + i] * iv);
                acc3 += vec4<f32>(weight_buf[wb3 + i] * iv);
            }
        }
    }

    let baseO = y * params.out_w * O + x * O;
    output_buf[baseO + o0] = act(acc0, params.activation);
    if (has1) { output_buf[baseO + o1] = act(acc1, params.activation); }
    if (has2) { output_buf[baseO + o2] = act(acc2, params.activation); }
    if (has3) { output_buf[baseO + o3] = act(acc3, params.activation); }
}
