enable f16;
// Per-channel scale: out[y, x, c] = x[y, x, c] * scale[c], scale is (1, 1, C).
// The broadcast multiply behind squeeze-excite / attention gates.

struct Params {
    n  : u32,   // H * W * channel groups (vec4 count)
    g  : u32,   // channel groups
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       scale_buf  : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    output_buf[i] = input_buf[i] * scale_buf[i % params.g];
}
