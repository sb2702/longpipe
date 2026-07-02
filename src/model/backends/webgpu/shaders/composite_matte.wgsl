// Render the raw 1-channel alpha matte as a premultiplied white silhouette
// (rgb = a, alpha = a) to the canvas swapchain. Doubles as a debug view AND a
// reusable mask a consumer can composite against their own source. Alpha only —
// no image, no background.
//
// Caller invariants (matched in CompositeMatteWebGPU):
//   - alpha is an NHWC vec4 storage buffer
//   - canvas.width === alpha.w, canvas.height === alpha.h (no resampling)

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
    width: u32,
};

@group(0) @binding(0) var<storage, read> alpha:  array<vec4<f32>>;
@group(0) @binding(1) var<uniform>       params: Params;

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    let x = u32(in.pos.x);
    let y = u32(in.pos.y);
    let i = y * params.width + x;
    let a = alpha[i].r;
    return vec4<f32>(a, a, a, a);
}
