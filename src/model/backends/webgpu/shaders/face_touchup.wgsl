// UV-space face touch-up (f32) — five passes, one module:
//   unwrap    : canonical mesh → 512² atlas; vertex at canonical UV, textured
//               by the frame at its landmark's position (vertex-pulled from the
//               landmark tensor, crop → frame via the box tensor). The
//               rasterizer's barycentric interpolation IS the piecewise-affine
//               warp — one draw call.
//   blur ×2   : separable gaussian on the atlas (freq-sep low band). Uses the
//               framebuffer-aligned vquad_fb so read row == write row and no
//               vertical flip accumulates across passes (WebGPU NDC-y is up,
//               texel rows are down).
//   combine   : low + (atlas − low) · detail — smooths skin, keeps pores.
//   composite : frame passthrough, then the mesh drawn at landmark screen
//               positions sampling the smoothed atlas, blended by the static
//               weight mask × strength. Score-gated: no face → passthrough.
//
// Landmarks are [0,1] CROP coords; the box tensor (cx, cy, halfSide/W, score,
// frame fractions) turns them into frame fractions in the vertex shaders —
// the whole path stays GPU-resident.

struct Uni {
    frame_w  : u32,
    frame_h  : u32,
    sigma    : f32,
    detail   : f32,
    strength : f32,
    thresh   : f32,
    dir      : vec2<f32>,   // blur step (1/atlas, 0) or (0, 1/atlas)
    canvas   : vec2<f32>,   // frame w, h (aspect for the box y half-extent)
    slots    : u32,         // faces packed into lm_buf / box_buf (1 or 4)
    grid     : u32,         // atlas tile grid: 1 → whole atlas, 2 → 2×2 tiles
}

@group(0) @binding(0) var<storage, read> frame_buf : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lm_buf    : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> box_buf   : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>       uni       : Uni;
@group(0) @binding(4) var samp : sampler;
@group(0) @binding(5) var tex0 : texture_2d<f32>;
@group(0) @binding(6) var tex1 : texture_2d<f32>;

// Bilinear sample of the frame tensor buffer at frame-fraction coords.
fn frame_at(px: i32, py: i32) -> vec3<f32> {
    let x = clamp(px, 0, i32(uni.frame_w) - 1);
    let y = clamp(py, 0, i32(uni.frame_h) - 1);
    return frame_buf[u32(y) * uni.frame_w + u32(x)].rgb;
}
fn frame_bilinear(f: vec2<f32>) -> vec3<f32> {
    let sx = clamp(f.x * f32(uni.frame_w) - 0.5, 0.0, f32(uni.frame_w - 1u));
    let sy = clamp(f.y * f32(uni.frame_h) - 0.5, 0.0, f32(uni.frame_h - 1u));
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let top = mix(frame_at(x0, y0),     frame_at(x0 + 1, y0),     tx);
    let bot = mix(frame_at(x0, y0 + 1), frame_at(x0 + 1, y0 + 1), tx);
    return mix(top, bot, ty);
}

// Landmark i → (frame-fraction x, y, score). halfSide is a fraction of frame
// WIDTH; the y half-extent rescales by the aspect (the box is px-square).
fn lm_frame(i: u32, face: u32) -> vec3<f32> {
    let box = box_buf[face];
    // lm_buf packs `slots` faces × 478 landmarks (ChannelConcat of the K
    // LandmarkNet outputs); 478 is even, so face f starts on a vec4 boundary.
    let gi = face * 478u + i;
    var g = lm_buf[gi / 2u];
    let lx = g[(gi % 2u) * 2u];
    let ly = g[(gi % 2u) * 2u + 1u];
    let hsx = box.z;
    let hsy = box.z * uni.canvas.x / uni.canvas.y;
    return vec3<f32>((box.x - hsx) + lx * 2.0 * hsx, (box.y - hsy) + ly * 2.0 * hsy, box.w);
}

// Face `face` owns tile (face % grid, face / grid) of the atlas. grid=1 (K=1)
// maps to the whole atlas — i.e. the single-face layout, unchanged.
fn tile_uv(uv: vec2<f32>, face: u32) -> vec2<f32> {
    let g = f32(uni.grid);
    return (uv + vec2<f32>(f32(face % uni.grid), f32(face / uni.grid))) / g;
}

// The tile containing `uv`, as (lo.x, lo.y, hi.x, hi.y) — filter passes clamp
// their taps to it so one face's skin can't bleed into a neighbour's tile.
fn tile_bounds(uv: vec2<f32>) -> vec4<f32> {
    let g = f32(uni.grid);
    let t = floor(clamp(uv, vec2<f32>(0.0), vec2<f32>(0.99999)) * g);
    return vec4<f32>(t / g, (t + vec2<f32>(1.0, 1.0)) / g);
}

// ── unwrap: mesh → atlas ──────────────────────────────────────────────────

struct UnwrapOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) src: vec2<f32>,   // frame-fraction sample position
};

@vertex
fn vs_unwrap(@builtin(instance_index) inst: u32,
             @location(0) a_uv: vec2<f32>, @location(1) a_idx: f32) -> UnwrapOut {
    var out: UnwrapOut;
    let l = lm_frame(u32(a_idx), inst);
    if (l.z < uni.thresh) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    out.src = l.xy;
    let uv = tile_uv(a_uv, inst);
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);   // atlas row == uv.y
    return out;
}

@fragment
fn fs_unwrap(in: UnwrapOut) -> @location(0) vec4<f32> {
    return vec4<f32>(frame_bilinear(in.src), 1.0);
}

// ── fullscreen quads ──────────────────────────────────────────────────────

struct QOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Framebuffer-aligned uv for FILTER passes: uv.y equals the output texel's
// row so read == write and no flip accumulates.
@vertex
fn vs_quad_fb(@builtin(vertex_index) vi: u32) -> QOut {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: QOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return out;
}

@fragment
fn fs_blur(in: QOut) -> @location(0) vec4<f32> {
    // dir is always ±1/ATLAS, so this recovers the atlas texel size.
    let tb = tile_bounds(in.uv);
    let ht = max(uni.dir.x, uni.dir.y) * 0.5;
    let lo = tb.xy + ht;
    let hi = tb.zw - ht;
    let s = max(uni.sigma, 0.001);
    let R = i32(clamp(ceil(s * 2.5), 1.0, 48.0));
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = -48; i <= 48; i++) {
        if (i < -R || i > R) { continue; }
        let w = exp(-f32(i * i) / (2.0 * s * s));
        sum += textureSample(tex0, samp, clamp(in.uv + uni.dir * f32(i), lo, hi)) * w;
        wsum += w;
    }
    return sum / wsum;
}

@fragment
fn fs_combine(in: QOut) -> @location(0) vec4<f32> {
    let a  = textureSample(tex0, samp, in.uv);
    let lo = textureSample(tex1, samp, in.uv);
    return clamp(lo + (a - lo) * uni.detail, vec4<f32>(0.0), vec4<f32>(1.0));
}

// Single-pass edge-preserving bilateral (style 'bilateral'). uni.dir carries
// the 2D texel size here; range sigma 0.15 (edge stop), spatial radius ≤ 12.
@fragment
fn fs_bilateral(in: QOut) -> @location(0) vec4<f32> {
    let btb = tile_bounds(in.uv);
    let bht = max(uni.dir.x, uni.dir.y) * 0.5;
    let blo = btb.xy + bht;
    let bhi = btb.zw - bht;
    let ss = max(uni.sigma, 0.001);
    let R = i32(clamp(ceil(ss), 1.0, 12.0));
    let sr = 0.15;
    let c = textureSample(tex0, samp, in.uv).rgb;
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var y = -12; y <= 12; y++) {
        if (y < -R || y > R) { continue; }
        for (var x = -12; x <= 12; x++) {
            if (x < -R || x > R) { continue; }
            let sc = textureSample(tex0, samp, clamp(in.uv + uni.dir * vec2<f32>(f32(x), f32(y)), blo, bhi)).rgb;
            let ws = exp(-f32(x * x + y * y) / (2.0 * ss * ss));
            let d = sc - c;
            let wr = exp(-dot(d, d) / (2.0 * sr * sr));
            sum += sc * ws * wr;
            wsum += ws * wr;
        }
    }
    return vec4<f32>(sum / wsum, 1.0);
}

// ── composite: passthrough + mesh over the face ───────────────────────────

@vertex
fn vs_pass(@builtin(vertex_index) vi: u32) -> QOut {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: QOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = vec2<f32>(0.0);
    return out;
}

@fragment
fn fs_pass(in: QOut) -> @location(0) vec4<f32> {
    let i = u32(in.pos.y) * uni.frame_w + u32(in.pos.x);
    return vec4<f32>(frame_buf[i].rgb, 1.0);
}

struct CompOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,      // TILED atlas uv — the atlas is per-face
    @location(1) src: vec2<f32>,     // frame-fraction position
    @location(2) uv_mask: vec2<f32>, // UNTILED canonical uv — the weight mask is
                                     // one static 512² asset shared by every face
};

@vertex
fn vs_comp(@builtin(instance_index) inst: u32,
           @location(0) a_uv: vec2<f32>, @location(1) a_idx: f32) -> CompOut {
    var out: CompOut;
    let l = lm_frame(u32(a_idx), inst);
    if (l.z < uni.thresh) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    out.uv = tile_uv(a_uv, inst);
    out.uv_mask = a_uv;
    out.src = l.xy;
    out.pos = vec4<f32>(l.x * 2.0 - 1.0, 1.0 - 2.0 * l.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_comp(in: CompOut) -> @location(0) vec4<f32> {
    let orig = frame_bilinear(in.src);
    let sm   = textureSample(tex0, samp, in.uv).rgb;                   // smoothed atlas — TILED
    // The weight mask is NOT tiled: it's one canonical 512² asset, so it must be
    // sampled with the untiled uv. Using the tiled uv here made every face read
    // only its quadrant OF THE MASK, stretched over the whole face — invisible at
    // slots=1 (tile_uv is identity) and badly wrong at slots=4.
    let w    = textureSample(tex1, samp, in.uv_mask).r * uni.strength; // weight mask — UNTILED
    return vec4<f32>(mix(orig, sm, w), 1.0);
}

// ── stage mode: unpack the composited float texture into the output tensor ──
// (WebGPU tensors are storage buffers; the mesh passes need a raster target.
// copyTextureToBuffer has a 256-byte bytesPerRow constraint that arbitrary
// widths violate, so a tiny compute blit does the copy instead.)

@group(0) @binding(7) var<storage, read_write> unpack_out : array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn cs_unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.frame_w || gid.y >= uni.frame_h) { return; }
    let v = textureLoad(tex0, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
    unpack_out[gid.y * uni.frame_w + gid.x] = v;
}
