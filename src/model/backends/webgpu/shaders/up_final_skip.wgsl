// up_final_skip: C/D alpha head. Fused concat(u, d_full, rgb) → conv 3×3 9→1
// → sigmoid. u = c_up=2 (.xy); d_full = c_high=4 full-res skip (full vec4);
// rgb = x_hr (.xyz). Channel order concat = 2 + 4 + 3 = 9. Output .x = alpha.
// Weight: 27 vec4 (3 per kpos): [kpos*3+0]=(w0,w1,0,0) u; [kpos*3+1]=(w2..w5)
// d_full; [kpos*3+2]=(w6,w7,w8,0) rgb. Bias .x.
// Binding order: u(0), d_full(1), rgb(2), weights(3), bias(4), params(5), output(6)

struct Params { h: u32, w: u32, _pad0: u32, _pad1: u32 }

@group(0) @binding(0) var<storage, read>       u_gru      : array<vec4<f32>>;   // .xy
@group(0) @binding(1) var<storage, read>       d_full     : array<vec4<f32>>;   // full vec4
@group(0) @binding(2) var<storage, read>       rgb        : array<vec4<f32>>;   // .xyz
@group(0) @binding(3) var<storage, read>       weight_buf : array<vec4<f32>>;   // 27 vec4
@group(0) @binding(4) var<storage, read>       bias_buf   : array<vec4<f32>>;   // .x
@group(0) @binding(5) var<uniform>             params     : Params;
@group(0) @binding(6) var<storage, read_write> output_buf : array<vec4<f32>>;

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
            acc += dot(weight_buf[kpos * 3u + 0u].xy,  u_gru[p].xy);
            acc += dot(weight_buf[kpos * 3u + 1u],     d_full[p]);
            acc += dot(weight_buf[kpos * 3u + 2u].xyz, rgb[p].xyz);
        }
    }

    output_buf[y * params.w + x] = vec4<f32>(1.0 / (1.0 + exp(-acc)), 0.0, 0.0, 0.0);
}
