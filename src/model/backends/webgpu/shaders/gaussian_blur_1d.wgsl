// Separable 1D Gaussian blur — run twice (h then v) for a 2D blur.
// Compile-time radius; sigma + direction are runtime uniforms.
// Tensor layout: NHWC vec4, index = y*W*(C/4) + x*(C/4) + c_group.

struct Params {
    in_w           : u32,
    in_h           : u32,
    channel_groups : u32,
    step_x         : i32,
    step_y         : i32,
    sigma          : f32,
    _pad0          : u32,
    _pad1          : u32,
}

@group(0) @binding(0) var<storage, read>       input_buf  : array<vec4<f32>>;
@group(0) @binding(1) var<uniform>             params     : Params;
@group(0) @binding(2) var<storage, read_write> output_buf : array<vec4<f32>>;

const R: i32 = 32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if (x >= params.in_w || y >= params.in_h || c >= params.channel_groups) {
        return;
    }

    let W = params.in_w;
    let H = params.in_h;
    let C = params.channel_groups;

    let two_s2 = 2.0 * params.sigma * params.sigma;

    var acc:  vec4<f32> = vec4<f32>(0.0);
    var wsum: f32       = 0.0;

    for (var i: i32 = -R; i <= R; i = i + 1) {
        let w  = exp(-f32(i * i) / two_s2);
        let sx = u32(clamp(i32(x) + i * params.step_x, 0, i32(W) - 1));
        let sy = u32(clamp(i32(y) + i * params.step_y, 0, i32(H) - 1));
        acc  = acc + w * input_buf[sy * W * C + sx * C + c];
        wsum = wsum + w;
    }

    output_buf[y * W * C + x * C + c] = acc / wsum;
}
