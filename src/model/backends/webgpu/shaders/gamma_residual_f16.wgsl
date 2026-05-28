enable f16;

// Gamma residual — full f16 variant.

struct Params {
    h               : u32,
    w               : u32,
    channel_groups  : u32,
    _pad0           : u32,
}

@group(0) @binding(0) var<storage, read>       b_buf      : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       h_new      : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read>       gamma      : array<vec4<f16>>;
@group(0) @binding(3) var<uniform>             params     : Params;
@group(0) @binding(4) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;
    if (x >= params.w || y >= params.h || c >= params.channel_groups) { return; }

    let idx = (y * params.w + x) * params.channel_groups + c;
    output_buf[idx] = b_buf[idx] + gamma[c] * h_new[idx];
}
