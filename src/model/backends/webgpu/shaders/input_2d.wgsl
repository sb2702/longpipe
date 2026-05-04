// Input op — sample a regular 2D source texture (RGBA8 unorm) into the NHWC
// vec4<f32> output buffer at the target resolution. Used for ImageBitmap
// sources, which are uploaded into a persistent staging texture via
// copyExternalImageToTexture before each dispatch.

struct Params {
    out_w : u32,
    out_h : u32,
    _pad0 : u32,
    _pad1 : u32,
}

@group(0) @binding(0) var                       src_tex     : texture_2d<f32>;
@group(0) @binding(1) var                       src_sampler : sampler;
@group(0) @binding(2) var<uniform>              params      : Params;
@group(0) @binding(3) var<storage, read_write>  output_buf  : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.out_w || y >= params.out_h) { return; }

    let uv = vec2<f32>(
        (f32(x) + 0.5) / f32(params.out_w),
        (f32(y) + 0.5) / f32(params.out_h),
    );
    let rgba = textureSampleLevel(src_tex, src_sampler, uv, 0.0);
    output_buf[y * params.out_w + x] = rgba;
}
