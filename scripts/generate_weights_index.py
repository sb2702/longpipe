#!/usr/bin/env python3
"""Generate manifest.json + index.html for a directory of model weight files.

Scans for model_*.bin (and .f16.bin) files, computes size and sha256 for each,
and writes both a machine-readable JSON manifest and a human-readable HTML
listing into the output directory. Designed for the CDN bucket layout — drop
the resulting files next to the weights to expose a browsable + scriptable
index at e.g. https://cdn.longpipe.dev/models/v/0.0.4/.

Usage:
    python scripts/generate_weights_index.py
    python scripts/generate_weights_index.py --src weights --out weights
    python scripts/generate_weights_index.py --base-url https://cdn.longpipe.dev/models/v/0.0.4/
"""

import argparse
import datetime
import hashlib
import json
import struct
from pathlib import Path

PATTERN = "model_*.bin"
# Audio-denoise assets live in the same CDN dir but aren't model_*.bin and have
# no SDK header (wasm is wasm; the dfn pack is its own format) — indexed as
# opaque binaries (size + sha256 only).
AUDIO_FILES = ("dfn.wasm", "rnnoise.wasm", "dfn_weights.pack", "dfn_weights_int8.pack")
# Touch-up static assets (the landmark weights already match model_*.bin;
# these two are the canonical face mesh + skin-weight mask).
TOUCHUP_FILES = ("face_topology.json", "weight_mask.png")
# Touch-up static assets (landmark weights match model_*.bin; these two don't).
TOUCHUP_FILES = ("face_topology.json", "weight_mask.png")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SRC = SCRIPT_DIR.parent / "weights"


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_header(path: Path) -> dict:
    """Parse the .bin header (see WEIGHTS_FORMAT.md): [u32 header_len][JSON][payload].
    Returns {dtype, kind, hasGru, top} describing the weight structure without
    reading the payload. Composite tier .bins have top-level base/wrapper(/gru);
    legacy base-only .bins have encoder/bottleneck/decoder."""
    try:
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<I", f.read(4))
            hdr = json.loads(f.read(hlen).decode("utf-8"))
    except Exception as e:
        return {"dtype": "?", "kind": "unreadable", "hasGru": False, "top": [], "error": str(e)}
    dtype = hdr.get("__dtype__", "f32")
    keys = sorted(k for k in hdr.keys() if k != "__dtype__")
    if "base" in keys and "wrapper" in keys:
        kind = "tier"
    elif "encoder" in keys:
        kind = "base-only"
    else:
        kind = "unknown"
    return {"dtype": dtype, "kind": kind, "hasGru": "gru" in keys, "top": keys}


def describe(path: Path) -> dict:
    """Structure metadata for video model_*.bin (via the SDK header); opaque
    metadata for the audio denoise assets (no header — size + sha256 only)."""
    if path.suffix == ".bin" and path.name.startswith("model_"):
        return read_header(path)
    kind = {".wasm": "audio-wasm", ".pack": "audio-weights",
            ".json": "touchup-asset", ".png": "touchup-asset"}.get(path.suffix, "audio")
    return {"dtype": "-", "kind": kind, "hasGru": False, "top": []}


def render_html(files, generated, base_url):
    rows = "\n".join(
        f'    <tr>'
        f'<td><a href="{f["name"]}">{f["name"]}</a></td>'
        f'<td>{human_size(f["size"])}</td>'
        f'<td><code>{f.get("dtype", "?")}</code></td>'
        f'<td>{f.get("kind", "?")}{" · gru" if f.get("hasGru") else ""}</td>'
        f'<td><code title="{f["sha256"]}">{f["sha256"][:12]}…</code></td>'
        f'</tr>'
        for f in files
    )
    base_line = f' · base URL <code>{base_url}</code>' if base_url else ''
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Longpipe weights</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 56rem; margin: 2rem auto; padding: 0 1rem; color: #222; }}
    h1 {{ margin-bottom: 0.25rem; }}
    p.lede {{ color: #555; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
    th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; }}
    th {{ font-weight: 600; color: #444; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; color: #555; }}
    a {{ color: #0a58ca; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    footer {{ margin-top: 2rem; color: #888; font-size: 0.85em; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #111; color: #ddd; }}
      p.lede, footer, th, code {{ color: #aaa; }}
      th, td {{ border-color: #2a2a2a; }}
      a {{ color: #6ea8fe; }}
    }}
  </style>
</head>
<body>
  <h1>Longpipe weights 🐉</h1>
  <p class="lede">Pre-trained model weights for the <a href="https://longpipe.dev">Longpipe</a> SDK. License: <a href="WEIGHTS_LICENSE">WEIGHTS_LICENSE</a> (MIT). Machine-readable index: <a href="manifest.json">manifest.json</a>.</p>
  <table>
    <thead>
      <tr><th>File</th><th>Size</th><th>dtype</th><th>format</th><th>SHA-256</th></tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  <footer>Generated {generated}{base_line}</footer>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, type=Path, help=f"Directory containing model_*.bin files (default: {DEFAULT_SRC})")
    ap.add_argument("--out", default=None, type=Path, help="Output directory for manifest.json + index.html (default: same as --src)")
    ap.add_argument("--base-url", default="", help="Optional base URL shown in the HTML footer")
    args = ap.parse_args()

    src = args.src.resolve()
    out = (args.out or args.src).resolve()
    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(src.glob(PATTERN)) + [src / n for n in AUDIO_FILES + TOUCHUP_FILES if (src / n).exists()]
    if not paths:
        raise SystemExit(f"No files matching {PATTERN} (or audio assets) in {src}")

    files = []
    for p in paths:
        size = p.stat().st_size
        digest = sha256_of(p)
        hdr = describe(p)
        files.append({"name": p.name, "size": size, "sha256": digest,
                      "dtype": hdr["dtype"], "kind": hdr["kind"], "hasGru": hdr["hasGru"]})
        gru = " gru" if hdr["hasGru"] else ""
        print(f"  {p.name}: {human_size(size)}  {hdr['dtype']} {hdr['kind']}{gru}  {digest[:12]}…")

    generated = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps({"generated": generated, "files": files}, indent=2) + "\n")
    print(f"wrote {manifest_path}")

    html_path = out / "index.html"
    html_path.write_text(render_html(files, generated, args.base_url))
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
