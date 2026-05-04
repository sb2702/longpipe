enable f16;

// Passthrough "compositor" — f16 variant. Image is stored as f16; values
// promote to f32 on read and the fragment writes f32 to the canvas
// swapchain (color attachment is the swapchain's preferred f32-equivalent
// format). See composite_passthrough.wgsl for the f32 version.

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

@group(0) @binding(0) var<storage, read> image:  array<vec4<f16>>;
@group(0) @binding(1) var<uniform>       params: Params;

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    let x = u32(in.pos.x);
    let y = u32(in.pos.y);
    let i = y * params.width + x;
    return vec4<f32>(vec3<f32>(image[i].rgb), 1.0);
}
