enable f16;

// up_final — full f16 variant. See up_final.wgsl for layout details.
// Fused concat(u, rgb) → conv 3×3 5→1 → sigmoid (A/B alpha head).
// Binding order: u(0), rgb(1), weights(2), bias(3), params(4), output(5)

struct Params { h: u32, w: u32, _pad0: u32, _pad1: u32 }

@group(0) @binding(0) var<storage, read>       u_gru      : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       rgb        : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read>       weight_buf : array<vec4<f16>>;
@group(0) @binding(3) var<storage, read>       bias_buf   : array<vec4<f16>>;
@group(0) @binding(4) var<uniform>             params     : Params;
@group(0) @binding(5) var<storage, read_write> output_buf : array<vec4<f16>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= params.w || y >= params.h) { return; }

    var acc = bias_buf[0].x;
    for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
            let iy = i32(y + ky) - 1; let ix = i32(x + kx) - 1;
            if (iy < 0 || ix < 0 || u32(iy) >= params.h || u32(ix) >= params.w) { continue; }
            let kpos = ky * 3u + kx;
            let p = u32(iy) * params.w + u32(ix);
            acc += dot(weight_buf[kpos].xy,       u_gru[p].xy);
            acc += dot(weight_buf[9u + kpos].xyz, rgb[p].xyz);
        }
    }

    output_buf[y * params.w + x] = vec4<f16>(1.0h / (1.0h + exp(-acc)), 0.0h, 0.0h, 0.0h);
}
