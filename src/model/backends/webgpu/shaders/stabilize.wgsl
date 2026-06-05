// Flow-gated temporal stabilizer (f32). See stabilize_f16.wgsl for the math.

struct Params {
    h        : u32,
    w        : u32,
    t_lo     : f32,
    t_hi     : f32,
    leak     : f32,
    release  : f32,
    t_div    : f32,
    div_scale: f32,
    step_x   : u32,
    step_y   : u32,
}

@group(0) @binding(0) var<storage, read>       flow_buf     : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       pred_buf     : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       ref_buf      : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       env_prev_buf : array<vec4<f32>>;
@group(0) @binding(4) var<uniform>             params       : Params;
@group(0) @binding(5) var<storage, read_write> output_buf   : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.w || y >= params.h) { return; }
    let idx = y * params.w + x;

    let mag      = length(flow_buf[idx].xy);
    let env_prev = env_prev_buf[idx].y;
    let env      = max(mag, params.release * env_prev);

    var g = clamp((env - params.t_lo) / max(params.t_hi - params.t_lo, 1e-3), 0.0, 1.0);
    g = max(g, params.leak);

    let pred = pred_buf[idx].x;
    let refv = ref_buf[idx].x;
    let stab = g * pred + (1.0 - g) * refv;

    output_buf[idx] = vec4<f32>(stab, env, 0.0, 0.0);
}
