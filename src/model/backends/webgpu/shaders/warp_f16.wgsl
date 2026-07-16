enable f16;

// Bilinear gather-warp (f16 storage, f32 coordinate math). For each output pixel
// p, sample the source at p + flow_scale·flow[p].xy and bilinearly interpolate,
// clamping the sample to the edge (border-replicate). Source + flow are 4-ch
// (1 group), same resolution; flow vector is in .xy.

struct Params {
    h          : u32,
    w          : u32,
    flow_scale : f32,
    groups     : u32,   // source/output channel groups (c/4); flow is always 1 group
}

@group(0) @binding(0) var<storage, read>       source_buf : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       flow_buf   : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             params     : Params;
@group(0) @binding(3) var<storage, read_write> output_buf : array<vec4<f16>>;

fn samp(x: i32, y: i32, W: i32, H: i32, g: u32) -> vec4<f32> {
    let cx = clamp(x, 0, W - 1);
    let cy = clamp(y, 0, H - 1);
    return vec4<f32>(source_buf[(u32(cy) * u32(W) + u32(cx)) * params.groups + g]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let g = gid.z;
    if (x >= params.w || y >= params.h || g >= params.groups) { return; }

    let W = i32(params.w);
    let H = i32(params.h);
    

    let f  = vec2<f32>(flow_buf[y * params.w + x].xy);
    let sx = clamp(f32(x) + params.flow_scale * f.x, 0.0, f32(W - 1));
    let sy = clamp(f32(y) + params.flow_scale * f.y, 0.0, f32(H - 1));

    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);

    let top = mix(samp(x0, y0, W, H, g), samp(x0 + 1, y0, W, H, g), tx);
    let bot = mix(samp(x0, y0 + 1, W, H, g), samp(x0 + 1, y0 + 1, W, H, g), tx);
    output_buf[(y * params.w + x) * params.groups + g] = vec4<f16>(mix(top, bot, ty));
}
