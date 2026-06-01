// cand_update_fused: ConvGRU candidate path + state update + output, fused.
// Production config (c_up=2, recurrent=1):
//   u_in      : c_up=2 (.x = a passthrough, .y = b recurrent)
//   h_prev    : hidden carrier — recurrent state in .z (the previous frame's
//               output of THIS op; on frame 0 it is a zero tensor)
//   gates_out : (.x = z, .y = r) from gates_fused
// cand_pre = bias + Σ_kpos (b_w*b_n + rh_w*(r_n*h_n));  h_til = tanh(cand_pre);
//   h_new = (1-z)*h_prev + z*h_til;  b_out = b + gamma*h_new.
// Output: vec4(a, b_out, h_new, 0) — .xy is the c_up=2 feature consumed
// downstream; .z carries h_new so the same tensor is fed back as next h_prev
// (no separate hidden-state buffer). Cand weight: 9 vec4 per kpos, .xy = (b_w, rh_w).
// Binding order: u_in(0), h_prev(1), gates_out(2), weight(3), bias(4), gamma(5), params(6), output(7)

struct Params { h: u32, w: u32, _pad0: u32, _pad1: u32 }

@group(0) @binding(0) var<storage, read>       u_in_buf      : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       h_prev_buf    : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       gates_out_buf : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       weight_buf    : array<vec4<f32>>;   // 9 vec4
@group(0) @binding(4) var<storage, read>       bias_buf      : array<vec4<f32>>;   // .x = cand_bias
@group(0) @binding(5) var<storage, read>       gamma_buf     : array<vec4<f32>>;   // .x = gamma
@group(0) @binding(6) var<uniform>             params        : Params;
@group(0) @binding(7) var<storage, read_write> output_buf    : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x; let y = gid.y;
    if (x >= params.w || y >= params.h) { return; }

    var cand_pre = bias_buf[0].x;
    for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
            let iy = i32(y + ky) - 1; let ix = i32(x + kx) - 1;
            if (iy < 0 || ix < 0 || u32(iy) >= params.h || u32(ix) >= params.w) { continue; }
            let kpos = ky * 3u + kx;
            let idx  = u32(iy) * params.w + u32(ix);
            let b_n  = u_in_buf[idx].y;
            let h_n  = h_prev_buf[idx].z;
            let r_n  = gates_out_buf[idx].y;
            let w    = weight_buf[kpos].xy;
            cand_pre += w.x * b_n + w.y * (r_n * h_n);
        }
    }

    let h_til      = tanh(cand_pre);
    let cur        = y * params.w + x;
    let u_cur      = u_in_buf[cur];
    let z_cur      = gates_out_buf[cur].x;
    let h_prev_cur = h_prev_buf[cur].z;
    let h_new      = (1.0 - z_cur) * h_prev_cur + z_cur * h_til;
    let b_out      = u_cur.y + gamma_buf[0].x * h_new;
    output_buf[cur] = vec4<f32>(u_cur.x, b_out, h_new, 0.0);
}
