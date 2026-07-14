enable f16;

// Landmark overlay (f16 storage, f32 math) — draws the image, then VERTEX-PULLS landmark dots
// straight from the LandmarkNet output tensor: the vertex shader fetches
// landmark i from the storage buffer and transforms crop coords → canvas NDC
// via the box tensor. The per-frame landmark data never touches the CPU.
//
// Entry points: vs_img/fs_img (fullscreen image, 6 verts) and vs_pts/fs_pts
// (count × 6 verts — one quad per landmark; WebGPU has no point size).

struct Params {
    img_w      : u32,
    count      : u32,
    thresh     : f32,   // hide all dots when box score < thresh
    point_size : f32,   // dot diameter in canvas px
    color      : vec4<f32>,
    canvas     : vec4<f32>,   // .xy = canvas w, h
}

@group(0) @binding(0) var<storage, read> image_buf : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> lm_buf    : array<vec4<f16>>;   // 478 × (x, y), [0,1] crop coords
@group(0) @binding(2) var<storage, read> box_buf   : array<vec4<f16>>;   // (cx, cy, halfSide/W, score) frame fractions
@group(0) @binding(3) var<uniform>       params    : Params;

// ── image pass ────────────────────────────────────────────────────────────

struct VOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_img(@builtin(vertex_index) vi: u32) -> VOut {
    let verts = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    var out: VOut;
    out.pos = vec4<f32>(verts[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_img(in: VOut) -> @location(0) vec4<f32> {
    let i = u32(in.pos.y) * params.img_w + u32(in.pos.x);
    return vec4<f32>(vec3<f32>(image_buf[i].rgb), 1.0);
}

// ── landmark dots ─────────────────────────────────────────────────────────

@vertex
fn vs_pts(@builtin(vertex_index) vi: u32) -> VOut {
    let i      = vi / 6u;
    let corner = vi % 6u;
    var out: VOut;

    let box = vec4<f32>(box_buf[0]);
    if (box.w < params.thresh || i >= params.count) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);   // clipped away
        return out;
    }

    // Landmark i: two (x, y) pairs per vec4.
    var g = vec4<f32>(lm_buf[i / 2u]);
    let lx = g[(i % 2u) * 2u];
    let ly = g[(i % 2u) * 2u + 1u];

    // Crop coords [0,1] → frame fractions. halfSide is a fraction of WIDTH;
    // the y half-extent rescales by the canvas aspect (the box is px-square).
    let hsx = box.z;
    let hsy = box.z * params.canvas.x / params.canvas.y;
    let px = (box.x - hsx) + lx * 2.0 * hsx;
    let py = (box.y - hsy) + ly * 2.0 * hsy;

    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    );
    let off = corners[corner] * params.point_size / params.canvas.xy;   // NDC half-extents ×2
    out.pos = vec4<f32>(px * 2.0 - 1.0 + off.x, 1.0 - 2.0 * py + off.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_pts(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(params.color.rgb, 1.0);
}
