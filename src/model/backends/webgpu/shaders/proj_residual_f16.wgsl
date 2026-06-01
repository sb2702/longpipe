enable f16;

// proj_residual — full f16 variant. See proj_residual.wgsl for layout details.
// Bespoke 1×1 conv (no activation) + residual add, fused.
// Binding order: input(0), skip(1), weights(2), bias(3), params(4), output(5)

struct Params {
    h          : u32,
    w          : u32,
    in_groups  : u32,
    out_groups : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       skip_buf   : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read>       weight_buf : array<mat4x4<f16>>;
@group(0) @binding(3) var<storage, read>       bias_buf   : array<vec4<f16>>;
@group(0) @binding(4) var<uniform>             params     : Params;
@group(0) @binding(5) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let o = gid.z;

    if (x >= params.w || y >= params.h || o >= params.out_groups) {
        return;
    }

    let I = params.in_groups;
    let O = params.out_groups;
    let pix = y * params.w + x;

    var result = bias_buf[o];
    for (var i = 0u; i < I; i++) {
        result += weight_buf[o * I + i] * input_buf[pix * I + i];
    }
    result += skip_buf[pix * O + o];

    output_buf[pix * O + o] = result;
}
