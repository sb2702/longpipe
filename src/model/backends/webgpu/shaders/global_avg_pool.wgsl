// Global average pool: (H, W, C) → (1, 1, C). One 256-thread workgroup per
// channel group; each thread sums a strided slice of the plane, then the
// partials are tree-reduced in workgroup memory. Accumulates in f32 for both
// storage dtypes. Ported from the webrtcHacks selfie-segmentation demo (a
// single thread walking the plane was 80% of that network's GPU time).

struct Params {
    n  : u32,   // H * W
    g  : u32,   // channel groups
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f32>>;

var<workgroup> partial : array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
    let c = wg.x;
    var sum = vec4<f32>(0.0);
    for (var i = t; i < params.n; i += 256u) { sum += input_buf[i * params.g + c]; }
    partial[t] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (t < stride) { partial[t] += partial[t + stride]; }
        workgroupBarrier();
    }
    if (t == 0u) { output_buf[c] = partial[0] / f32(params.n); }
}
