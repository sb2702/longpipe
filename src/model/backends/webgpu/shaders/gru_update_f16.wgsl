enable f16;

// GRU update — full f16 variant.

struct Params {
    size  : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       z_buf   : array<f16>;
@group(0) @binding(1) var<storage, read>       h_prev  : array<f16>;
@group(0) @binding(2) var<storage, read>       h_til   : array<f16>;
@group(0) @binding(3) var<uniform>             params  : Params;
@group(0) @binding(4) var<storage, read_write> h_new   : array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    let z = z_buf[idx];
    h_new[idx] = (f16(1.0) - z) * h_prev[idx] + z * h_til[idx];
}
