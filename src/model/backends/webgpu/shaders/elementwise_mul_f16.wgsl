enable f16;

// Element-wise multiply — full f16 variant.
// array<f16> is binary-compatible with array<vec4<f16>> written by the f16 conv shaders.

struct Params {
    size  : u32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}

@group(0) @binding(0) var<storage, read>       input_a : array<f16>;
@group(0) @binding(1) var<storage, read>       input_b : array<f16>;
@group(0) @binding(2) var<uniform>             params  : Params;
@group(0) @binding(3) var<storage, read_write> output  : array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
    output[idx] = input_a[idx] * input_b[idx];
}
