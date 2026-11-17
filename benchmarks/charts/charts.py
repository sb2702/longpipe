#!/usr/bin/env python3
"""Render the README charts (horizontal bar SVGs, no dependencies).

    python3 benchmarks/charts/charts.py   # writes benchmarks/charts/speed-360p.svg and quality-p3m.svg

Numbers come from the results files linked in benchmarks/README.md; edit ROWS below when they change.
Style follows the selfie-segmentation chart in github.com/sb2702/webrtc-hacks-webgpu.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
LONGPIPE, MEDIAPIPE = "#2a78d6", "#9a9890"

SPEED = dict(
    file="speed-360p.svg",
    title="Background replacement in the browser: frames per second",
    subtitle="360p decode → background replacement → H.264 encode · Apple M4, Chrome 157 · Longpipe 0.0.18 · higher is better",
    axis="frames per second", ticks=[0, 100, 200, 300, 400], fmt=lambda v: f"{v:.0f}",
    rows=[  # (label, value, colour)
        ("MediaPipe selfie on Longpipe (WebGPU)", 379, LONGPIPE),
        ("Longpipe xs (WebGPU)", 344, LONGPIPE),
        ("Longpipe small (WebGPU)", 332, LONGPIPE),
        ("Longpipe medium (WebGPU)", 250, LONGPIPE),
        ("Longpipe large (WebGPU)", 158, LONGPIPE),
        ("Longpipe xl (WebGPU)", 83, LONGPIPE),
        ("MediaPipe tasks-vision (GPU, WebGL)", 123, MEDIAPIPE),
    ],
)

QUALITY = dict(
    file="quality-p3m.svg",
    title="Matte error on P3M-500-P (mean absolute alpha error)",
    subtitle="P3M-10k validation, 500 portrait photos, scored at full resolution · lower is better",
    axis="mean absolute alpha error (MAD)", ticks=[0, 0.01, 0.02, 0.03, 0.04], fmt=lambda v: f"{v:.4f}", tickfmt=lambda v: f"{v:.2f}",
    rows=[
        ("Longpipe xl", 0.0071, LONGPIPE),
        ("Longpipe medium", 0.0106, LONGPIPE),
        ("Longpipe large", 0.0111, LONGPIPE),
        ("Longpipe small", 0.0170, LONGPIPE),
        ("Longpipe xs", 0.0210, LONGPIPE),
        ("MediaPipe selfie (square)", 0.0304, MEDIAPIPE),
        ("MediaPipe selfie (landscape)", 0.0322, MEDIAPIPE),
    ],
)


def render(c):
    W, x0, x1, top, row = 820, 330, 760, 72, 30
    n = len(c["rows"])
    plot_h = n * row + 6
    H = top + plot_h + 46
    vmax = c["ticks"][-1]
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" role="img" aria-labelledby="t d">',
         f'<title id="t">{escape(c["title"])}</title>',
         f'<desc id="d">{escape(c["subtitle"])}. ' + "; ".join(f"{escape(l)}: {c['fmt'](v)}" for l, v, _ in c["rows"]) + ".</desc>",
         f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
         f'<text x="16" y="30" font-size="18" font-weight="600" fill="#0b0b0b">{escape(c["title"])}</text>',
         f'<text x="16" y="52" font-size="13" fill="#52514e">{escape(c["subtitle"])}</text>']
    yb = top + plot_h
    for t in c["ticks"]:
        x = sx(t)
        o.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{yb}" stroke="#e6e5e0" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{yb + 18}" font-size="12" fill="#52514e" text-anchor="middle">{c.get("tickfmt", c["fmt"])(t) if t else 0}</text>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{yb + 36}" font-size="12" fill="#52514e" text-anchor="middle">{escape(c["axis"])}</text>')
    for i, (label, v, col) in enumerate(c["rows"]):
        y = top + 12 + i * row
        xe = sx(v)
        r = min(4, (xe - x0) / 2)
        o.append(f'<g><title>{escape(label)}: {c["fmt"](v)}</title><path d="M{x0},{y} H{xe - r:.1f} Q{xe:.1f},{y} {xe:.1f},{y + r} '
                 f'V{y + 18 - r} Q{xe:.1f},{y + 18} {xe - r:.1f},{y + 18} H{x0} Z" fill="{col}"/></g>')
        o.append(f'<text x="{x0 - 10}" y="{y + 13.5}" font-size="13" fill="#0b0b0b" text-anchor="end">{escape(label)}</text>')
        o.append(f'<text x="{xe + 6:.1f}" y="{y + 13.5}" font-size="12" fill="#52514e" font-variant-numeric="tabular-nums">{c["fmt"](v)}</text>')
    o.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{yb}" stroke="#52514e" stroke-width="1"/>')
    o.append("</svg>")
    (OUT / c["file"]).write_text("\n".join(o) + "\n")
    print("wrote", OUT / c["file"])


if __name__ == "__main__":
    render(SPEED)
    render(QUALITY)
