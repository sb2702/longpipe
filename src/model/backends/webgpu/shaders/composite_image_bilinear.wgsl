// Like composite_image but bg is sampled bilinearly — bg may be smaller than
// (image, alpha). Used by CompositorBlur to skip the final full-res upsample
// in the blur pyramid and let this shader's own per-pixel scan do the
// expansion (the work is already happening here for the composite anyway).
//
// Layout invariant: image, alpha share output dims (canvas h × w); bg is
// at a smaller resolution (bg_h × bg_w). All NHWC vec4 storage buffers,
// channelGroups = 1 (RGB padded to vec4).

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOut {
    let verts = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    var out: VertexOut;
    out.pos = vec4<f32>(verts[vi], 0.0, 1.0);
    return out;
}

struct Params {
    out_w: u32,
    out_h: u32,
    bg_w:  u32,
    bg_h:  u32,
};

@group(0) @binding(0) var<storage, read> image:  array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> alpha:  array<vec4<f32>>;
@group(0) @binding(2) var<uniform>       params: Params;
@group(0) @binding(3) var<storage, read> bg:     array<vec4<f32>>;

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    let x = u32(in.pos.x);
    let y = u32(in.pos.y);
    let i = y * params.out_w + x;

    let fg = image[i].rgb;
    let a  = alpha[i].r;

    // Bilinear sample bg at the corresponding location. align_corners=False.
    let src_x = (f32(x) + 0.5) * (f32(params.bg_w) / f32(params.out_w)) - 0.5;
    let src_y = (f32(y) + 0.5) * (f32(params.bg_h) / f32(params.out_h)) - 0.5;
    let x0 = u32(clamp(i32(floor(src_x)),     0, i32(params.bg_w) - 1));
    let x1 = u32(clamp(i32(floor(src_x)) + 1, 0, i32(params.bg_w) - 1));
    let y0 = u32(clamp(i32(floor(src_y)),     0, i32(params.bg_h) - 1));
    let y1 = u32(clamp(i32(floor(src_y)) + 1, 0, i32(params.bg_h) - 1));
    let wx = src_x - floor(src_x);
    let wy = src_y - floor(src_y);

    let tl = bg[y0 * params.bg_w + x0].rgb;
    let tr = bg[y0 * params.bg_w + x1].rgb;
    let bl = bg[y1 * params.bg_w + x0].rgb;
    let br = bg[y1 * params.bg_w + x1].rgb;
    let bgc = (1.0 - wy) * ((1.0 - wx) * tl + wx * tr)
            +        wy  * ((1.0 - wx) * bl + wx * br);

    let rgb = fg * a + bgc * (1.0 - a);
    return vec4<f32>(rgb, 1.0);
}
