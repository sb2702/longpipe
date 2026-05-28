// Element-wise tanh — used by ConvGRU candidate activation.
// Operates on packed vec4 buffers (NHWC layout). tanh() is element-wise on vec4.

struct Params {
    n_groups : u32,
    _pad0    : u32,
    _pad1    : u32,
    _pad2    : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_groups) { return; }
    output_buf[idx] = tanh(input_buf[idx]);
}
