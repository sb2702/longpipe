enable f16;

// gates_fused — full f16 variant. See gates_fused.wgsl for layout details.
// ConvGRU z + r gates (production config c_up=2, recurrent=1).
// Binding order: u_in(0), h_prev(1), weights(2), bias(3), params(4), output(5)

struct Params { h: u32, w: u32, _pad0: u32, _pad1: u32 }

@group(0) @binding(0) var<storage, read>       u_in_buf   : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       h_prev_buf : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read>       weight_buf : array<vec4<f16>>;
@group(0) @binding(3) var<storage, read>       bias_buf   : array<vec4<f16>>;
@group(0) @binding(4) var<uniform>             params     : Params;
@group(0) @binding(5) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= params.w || y >= params.h) { return; }

    let bias = bias_buf[0].xy;
    var z_pre = bias.x;
    var r_pre = bias.y;

    for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
            let iy = i32(y + ky) - 1; let ix = i32(x + kx) - 1;
            if (iy < 0 || ix < 0 || u32(iy) >= params.h || u32(ix) >= params.w) { continue; }
            let kpos = ky * 3u + kx;
            let idx  = u32(iy) * params.w + u32(ix);
            let b_n  = u_in_buf[idx].y;
            let h_n  = h_prev_buf[idx].z;
            let w    = weight_buf[kpos];
            z_pre += w.x * b_n + w.y * h_n;
            r_pre += w.z * b_n + w.w * h_n;
        }
    }

    let z = 1.0h / (1.0h + exp(-z_pre));
    let r = 1.0h / (1.0h + exp(-r_pre));
    output_buf[y * params.w + x] = vec4<f16>(z, r, 0.0h, 0.0h);
}
