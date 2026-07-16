#version 300 es
// Auto-reframe state update → 1×1 texture: (cx, cy, size, moving), frame
// fractions. See reframe_state.wgsl for the full commentary — this mirrors it.
// One fragment does the whole thing (it's one camera's worth of arithmetic).

precision highp float;
precision highp int;

uniform sampler2D u_box;    // 1×K — (cx, cy, halfSide/W, score)
uniform sampler2D u_prev;   // 1×1 — previous state
uniform sampler2D u_cmd;    // 1×1 — .x = mode
uniform int   u_k;
uniform float u_zoom;
uniform float u_gravity;
uniform float u_margin;
uniform float u_deadband;
uniform float u_ease;
uniform float u_aspect;     // frame w/h

out vec4 fragColor;

const float MODE_HOLD  = 1.0;
const float MODE_SOLVE = 2.0;

void main() {
    vec4 prev = texelFetch(u_prev, ivec2(0, 0), 0);
    float mode = texelFetch(u_cmd, ivec2(0, 0), 0).x;

    // Subject = largest live face (no hysteresis yet — see the WGSL note).
    int best = -1;
    float bestHs = 0.0;
    for (int i = 0; i < u_k; i++) {
        vec4 b = texelFetch(u_box, ivec2(i, 0), 0);
        if (b.w > 0.0 && b.z > bestHs) { bestHs = b.z; best = i; }
    }
    if (best < 0) { fragColor = prev; return; }
    vec4 subj = texelFetch(u_box, ivec2(best, 0), 0);

    float halfX = subj.z;
    float halfY = subj.z * u_aspect;
    // Fallback = the full frame (no reframe) — see reframe_state.wgsl.
    float wcx = 0.5, wcy = 0.5, ws = 1.0;
    for (int t = 0; t <= 25; t++) {
        float z = u_zoom + (1.0 - u_zoom) * (float(t) / 25.0);
        float s = 1.0 / z;
        float cx = clamp(0.5 + u_gravity * (subj.x - 0.5), s * 0.5, 1.0 - s * 0.5);
        float cy = clamp(0.5 + u_gravity * (subj.y - 0.5), s * 0.5, 1.0 - s * 0.5);
        if (abs(subj.x - cx) + halfX + u_margin <= s * 0.5 &&
            abs(subj.y - cy) + halfY + u_margin <= s * 0.5) {
            wcx = cx; wcy = cy; ws = s;
            break;
        }
    }
    if (prev.z <= 0.0 || mode == MODE_SOLVE) { fragColor = vec4(wcx, wcy, ws, 0.0); return; }
    if (mode == MODE_HOLD) { fragColor = prev; return; }

    float d = length(vec2(wcx - prev.x, wcy - prev.y)) + abs(ws - prev.z);
    float moving = prev.w;
    if (d > u_deadband) moving = 1.0;
    else if (d < u_deadband * 0.3) moving = 0.0;
    if (moving < 0.5) { fragColor = vec4(prev.xyz, 0.0); return; }

    fragColor = vec4(
        mix(prev.x, wcx, u_ease),
        mix(prev.y, wcy, u_ease),
        mix(prev.z, ws,  u_ease),
        1.0
    );
}
