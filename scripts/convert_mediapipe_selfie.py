#!/usr/bin/env python3
"""Convert Google's MediaPipe selfie segmenter (.tflite) to Longpipe's weight pack.

    pip install "tensorflow>=2.16,<2.20" numpy pillow flatbuffers
    python3 scripts/convert_mediapipe_selfie.py                 # download, convert, verify
    python3 scripts/convert_mediapipe_selfie.py --image me.png  # use a real frame for the check

What it does, per variant (square 256×256, landscape 144×256):

  1. Downloads the official float16 .tflite from storage.googleapis.com (pinned
     version, sha256-checked) — or reads it from --tflite-dir.
  2. Reads every conv / depthwise / transposed-conv layer's weights straight out
     of the flatbuffer (the fp16 constants behind each DEQUANTIZE).
  3. Repacks them into Longpipe's shader layout (the same packers the SDK's own
     tiers use — see WEIGHTS_FORMAT.md):
        conv / transposed conv  [O, kh, kw, I]  → mat4x4[kh*kw][O/4][I/4], column = in-channel
        depthwise               [1, kh, kw, C]  → vec4[kh*kw][C/4]
        bias                    [O]             → padded to a multiple of 4
  4. Writes `mediapipe_selfie_segmenter[_landscape].f16.bin` in Longpipe's binary
     pack format (uint32 header length + JSON header + payload; "__dtype__": "f16").
     The source weights ARE fp16, so the f16 pack is lossless (asserted below).
  5. Verifies: a numpy re-implementation of the network, run on the packed
     weights' source arrays, against the TFLite interpreter (max |diff| printed),
     and writes golden input/output pairs to tests/fixtures/local/ for the
     browser parity test (tests/networks/mediapipe_selfie.local.test.ts).
  6. Copies LICENSE (Apache-2.0) + NOTICE next to the weights, and writes a
     manifest.json with sizes and sha256s.

Output (default): dist-weights/mediapipe/selfie/v1/ — upload that directory to
https://cdn.longpipe.dev/models/mediapipe/selfie/v1/ (see third_party/mediapipe/NOTICE.md).

The network graph itself lives in src/model/networks/mediapipe_selfie.ts (model
as code); this script only moves weights. Ported from the webrtcHacks
segmentation demo (github.com/sb2702/webrtc-hacks-webgpu, MIT):
tools/tflite_graph.py, tools/export_weights.py, tools/selfie_numpy.py.
"""
import argparse, glob, hashlib, importlib.util, json, os, shutil, site, struct, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Pinned to version 1 (not `latest`) so the CDN files are reproducible.
SOURCES = {
    "square": dict(
        url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        sha256="191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
        out="mediapipe_selfie_segmenter.f16.bin", h=256, w=256),
    "landscape": dict(
        url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite",
        sha256="490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e",
        out="mediapipe_selfie_segmenter_landscape.f16.bin", h=144, w=256),
}
MODEL_CARD = "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf"
# The final transposed conv is called `segment` in the square model and
# `conv2d_transpose` in the landscape one; everything else matches 1:1.
RENAME = {"conv2d_transpose": "segment"}


# ── minimal .tflite reader (flatbuffers + TF's generated schema) ────────────

def _load_schema():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        hits = glob.glob(f"{sp}/tensorflow/lite/python/schema_py_generated.py")
        if hits:
            spec = importlib.util.spec_from_file_location("tfl_schema", hits[0])
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
    raise ImportError("tensorflow/lite/python/schema_py_generated.py not found (pip install tensorflow)")

S = _load_schema()
BUILTIN = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
PADDING = {S.Padding.SAME: "SAME", S.Padding.VALID: "VALID"}
DTYPE = {S.TensorType.FLOAT32: np.float32, S.TensorType.FLOAT16: np.float16, S.TensorType.INT32: np.int32}
KINDS = {"CONV_2D": "conv", "DEPTHWISE_CONV_2D": "depthwise", "Convolution2DTransposeBias": "conv_transpose"}


class Graph:
    def __init__(self, path):
        buf = open(path, "rb").read()
        self.model = S.Model.GetRootAsModel(buf, 0)
        sg = self.model.Subgraphs(0)
        self.tensors = [sg.Tensors(i) for i in range(sg.TensorsLength())]
        self._ops = []
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            code = self.model.OperatorCodes(op.OpcodeIndex())
            b = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            kind = code.CustomCode().decode() if b == S.BuiltinOperator.CUSTOM else BUILTIN[b]
            self._ops.append(dict(kind=kind, op=op,
                                  inputs=[op.Inputs(j) for j in range(op.InputsLength())],
                                  outputs=[op.Outputs(j) for j in range(op.OutputsLength())]))

    def name(self, t): return self.tensors[t].Name().decode()

    def raw(self, t):
        tensor = self.tensors[t]
        b = self.model.Buffers(tensor.Buffer())
        if b.DataLength() == 0: return None
        shape = tensor.ShapeAsNumpy().tolist()
        return np.frombuffer(b.DataAsNumpy().tobytes(), DTYPE[tensor.Type()]).reshape(shape)

    def const_f16(self, t):
        """Weights as stored (fp16), following a DEQUANTIZE producer if there is one."""
        d = self.raw(t)
        if d is None:
            for o in self._ops:
                if o["outputs"] == [t] and o["kind"] == "DEQUANTIZE":
                    d = self.raw(o["inputs"][0])
        assert d is not None and d.dtype == np.float16, f"tensor {t}: expected fp16 constant"
        return d

    def layers(self):
        """[(name, kind, w_fp16, b_fp16, pad, stride)] in graph order."""
        out = []
        for o in self._ops:
            kind = KINDS.get(o["kind"])
            if not kind: continue
            pad, stride = "VALID", 2   # transposed conv: k2 s2 VALID (custom op, no options)
            t = o["op"].BuiltinOptions()
            if o["kind"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                opt = (S.Conv2DOptions if o["kind"] == "CONV_2D" else S.DepthwiseConv2DOptions)()
                opt.Init(t.Bytes, t.Pos)
                assert opt.StrideH() == opt.StrideW()
                assert opt.FusedActivationFunction() == 0, "fused activations not expected"
                pad, stride = PADDING[opt.Padding()], opt.StrideH()
            name = self.name(o["outputs"][0])
            out.append((RENAME.get(name, name), kind,
                        self.const_f16(o["inputs"][1]), self.const_f16(o["inputs"][2]), pad, stride))
        return out


# ── packing into Longpipe's shader layout ───────────────────────────────────

def groups(c): return -(-c // 4)

def pack_conv(w):                      # [O, kh, kw, I] → [kh*kw][Og][Ig] of 4×4 (col = in, row = out)
    O, kh, kw, I = w.shape
    out = np.zeros((kh * kw, groups(O), groups(I), 4, 4), w.dtype)
    for o in range(O):
        for i in range(I):
            out[:, o // 4, i // 4, i % 4, o % 4] = w[o, :, :, i].reshape(-1)
    return out.ravel()

def pack_depthwise(w):                 # [1, kh, kw, C] → [kh*kw][Cg] of vec4
    _, kh, kw, C = w.shape
    out = np.zeros((kh * kw, groups(C) * 4), w.dtype)
    out[:, :C] = w.reshape(kh * kw, C)
    return out.ravel()

def pack_bias(b):
    out = np.zeros(groups(len(b)) * 4, b.dtype); out[:len(b)] = b
    return out


def write_pack(path, layers, meta):
    header, chunks, offset = {"__dtype__": "f16", "meta": meta, "layers": {}}, [], 0
    for name, kind, w, b, _, _ in layers:
        pw = pack_depthwise(w) if kind == "depthwise" else pack_conv(w)
        pb = pack_bias(b)
        entry = {}
        for key, arr in (("weights", pw), ("bias", pb)):
            entry[key] = {"offset": offset, "length": int(arr.size)}
            chunks.append(arr.astype(np.float16)); offset += arr.size
        header["layers"][name] = entry
    hdr = json.dumps(header, separators=(",", ":")).encode()
    hdr += b" " * ((-(4 + len(hdr))) % 4)          # payload 4-byte aligned
    payload = np.concatenate(chunks).astype("<f2").tobytes()
    Path(path).write_bytes(struct.pack("<I", len(hdr)) + hdr + payload)


def read_pack(path):
    """Mirror of src/utils/loadWeights.ts, for the lossless round-trip check."""
    buf = Path(path).read_bytes()
    n = struct.unpack("<I", buf[:4])[0]
    header = json.loads(buf[4:4 + n])
    payload = np.frombuffer(buf[4 + n:], "<f2")
    return {k: {kk: payload[r["offset"]:r["offset"] + r["length"]] for kk, r in v.items()}
            for k, v in header["layers"].items()}, header


# ── numpy reference network (mirrors src/model/networks/mediapipe_selfie.ts) ─

def act(x, a):
    if a == "relu": return np.maximum(x, 0)
    if a == "hswish": return x * np.clip(x + 3, 0, 6) / 6
    if a == "sigmoid": return 1 / (1 + np.exp(-x))
    return x

def same_pad(n, k, s):
    out = -(-n // s)
    return out, max((out - 1) * s + k - n, 0) // 2

def selfie(img, L):                    # img [H, W, 3] in [0, 1]; L: name → (kind, w, b, pad, stride)
    def conv(x, name, a="none"):
        _, w, b, _, s = L[name]; O, kh, kw, _ = w.shape
        H, W, _ = x.shape; oh, pt = same_pad(H, kh, s); ow, pl = same_pad(W, kw, s)
        xp = np.pad(x, ((pt, kh), (pl, kw), (0, 0)))
        y = np.zeros((oh, ow, O), np.float32) + b
        for ky in range(kh):
            for kx in range(kw):
                y += xp[ky:ky + oh * s:s, kx:kx + ow * s:s] @ w[:, ky, kx, :].T
        return act(y, a)
    def depthwise(x, name, a="none"):
        _, w, b, _, s = L[name]; _, kh, kw, C = w.shape
        H, W, _ = x.shape; oh, pt = same_pad(H, kh, s); ow, pl = same_pad(W, kw, s)
        xp = np.pad(x, ((pt, kh), (pl, kw), (0, 0)))
        y = np.zeros((oh, ow, C), np.float32) + b
        for ky in range(kh):
            for kx in range(kw):
                y += xp[ky:ky + oh * s:s, kx:kx + ow * s:s] * w[0, ky, kx]
        return act(y, a)
    def conv_t(x, name):
        _, w, b, _, _ = L[name]; H, W, _ = x.shape
        y = np.zeros((H * 2, W * 2, w.shape[0]), np.float32)
        for dy in range(2):
            for dx in range(2):
                y[dy::2, dx::2] = x @ w[:, dy, dx, :].T + b
        return y
    pool = lambda x: x.mean(axis=(0, 1), keepdims=True)
    def up2(x):
        H, W, _ = x.shape
        def idx(n):
            src = np.clip((np.arange(n * 2) + 0.5) / 2 - 0.5, 0, n - 1)
            i0 = np.floor(src).astype(int); return i0, np.minimum(i0 + 1, n - 1), (src - i0)[:, None]
        y0, y1, fy = idx(H); x0, x1, fx = idx(W)
        rows = x[y0] * (1 - fy[:, :, None]) + x[y1] * fy[:, :, None]
        return rows[:, x0] * (1 - fx[None]) + rows[:, x1] * fx[None]
    se = lambda x, a, b: x * conv(conv(pool(x), a, "relu"), b, "sigmoid")
    def ir(x, e, d, p, se_=None, a="relu", res=False):
        y = depthwise(conv(x, e, a), d, a)
        if se_: y = se(y, *se_)
        y = conv(y, p)
        return x + y if res else y
    def dec(x, skip, proj, s_, c1, dw):
        up = conv(up2(x), proj)
        g = conv(conv(pool(skip + up), s_[0], "relu"), s_[1], "sigmoid")
        y = conv(skip * g + up, c1, "relu")
        return y + depthwise(y, dw, "relu")
    s1 = conv(img, "conv2d", "hswish")
    x = depthwise(conv(s1, "conv2d_1", "relu"), "depthwise_conv2d", "relu")
    s2 = conv(se(x, "conv2d_2", "conv2d_3"), "conv2d_4")
    x = ir(s2, "conv2d_5", "depthwise_conv2d_1", "conv2d_6")
    s3 = ir(x, "conv2d_7", "depthwise_conv2d_2", "conv2d_8", res=True)
    x = ir(s3, "conv2d_9", "depthwise_conv2d_3", "conv2d_12", ("conv2d_10", "conv2d_11"), "hswish")
    for i in range(4):
        n = 13 + 4 * i
        x = ir(x, f"conv2d_{n}", f"depthwise_conv2d_{4 + i}", f"conv2d_{n + 3}",
               (f"conv2d_{n + 1}", f"conv2d_{n + 2}"), "hswish", True)
    x = conv(x, "conv2d_29", "relu") * conv(pool(x), "conv2d_30", "sigmoid")
    x = dec(x, s3, "conv2d_31", ("conv2d_32", "conv2d_33"), "conv2d_34", "depthwise_conv2d_8")
    x = dec(x, s2, "conv2d_35", ("conv2d_36", "conv2d_37"), "conv2d_38", "depthwise_conv2d_9")
    x = dec(x, s1, "conv2d_39", ("conv2d_40", "conv2d_41"), "conv2d_42", "depthwise_conv2d_10")
    return act(conv_t(x, "segment"), "sigmoid")[..., 0]


def test_image(h, w, path):
    if path:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255
    # Deterministic synthetic frame: smooth background + a head-and-shoulders blob.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    bg = np.stack([0.3 + 0.2 * u, 0.4 + 0.2 * v, 0.5 - 0.2 * u * v], -1)
    head = ((u - 0.5) / 0.14) ** 2 + ((v - 0.38) / 0.2) ** 2 < 1
    body = ((u - 0.5) / 0.32) ** 2 + ((v - 1.05) / 0.45) ** 2 < 1
    fg = (head | body)[..., None]
    img = np.where(fg, np.array([0.85, 0.65, 0.55], np.float32), bg)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)


def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "dist-weights/mediapipe/selfie/v1"))
    ap.add_argument("--tflite-dir", help="read <name>.tflite from here instead of downloading")
    ap.add_argument("--image", help="RGB image for the numpy-vs-TFLite check + test goldens (default: synthetic)")
    ap.add_argument("--fixtures", default=str(ROOT / "tests/fixtures/local"),
                    help="where to write golden input/output JSON for the browser parity test ('' to skip)")
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.tflite_dir) if args.tflite_dir else out / ".src"
    cache.mkdir(parents=True, exist_ok=True)
    names = None
    manifest = {"source": "MediaPipe Selfie Segmenter (Google), Apache-2.0",
                "model_card": MODEL_CARD, "files": {}}

    for variant, src in SOURCES.items():
        tfl = cache / Path(src["url"]).name
        if not tfl.exists():
            print(f"downloading {src['url']}")
            urllib.request.urlretrieve(src["url"], tfl)
        assert sha256(tfl) == src["sha256"], f"{tfl}: sha256 mismatch (upstream file changed?)"

        layers = Graph(str(tfl)).layers()
        ln = [(n, k, w.shape, p, s) for n, k, w, _, p, s in layers]
        if names is None: names = ln
        assert ln == names, f"{variant}: layer table differs from square model"

        meta = {"model": "mediapipe-selfie", "variant": variant, "input": {"w": src["w"], "h": src["h"]},
                "source": src["url"], "source_sha256": src["sha256"], "license": "Apache-2.0"}
        dst = out / src["out"]
        write_pack(dst, layers, meta)

        # Lossless: unpacking the pack reproduces exactly the packed fp16 source weights.
        packed, _ = read_pack(dst)
        for name, kind, w, b, _, _ in layers:
            pw = pack_depthwise(w) if kind == "depthwise" else pack_conv(w)
            assert np.array_equal(packed[name]["weights"], pw) and np.array_equal(packed[name]["bias"], pack_bias(b)), name

        # numpy network vs the TFLite interpreter, on the same input.
        L = {n: (k, w.astype(np.float32), b.astype(np.float32), p, s) for n, k, w, b, p, s in layers}
        img = test_image(src["h"], src["w"], args.image)
        ours = selfie(img, L)
        it = tf.lite.Interpreter(model_path=str(tfl)); it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], img[None])
        it.invoke()
        ref = it.get_tensor(it.get_output_details()[0]["index"])[0, ..., 0]
        diff = np.abs(ours - ref)
        print(f"{variant:9s} {src['w']}x{src['h']}  {len(layers)} layers  {dst.stat().st_size:,} B  "
              f"numpy vs tflite: max {diff.max():.2e} mean {diff.mean():.2e}  (mask mean {ref.mean():.3f})")
        assert diff.max() < 1e-4, "numpy reference diverges from TFLite"

        if args.fixtures:
            fx = Path(args.fixtures); fx.mkdir(parents=True, exist_ok=True)
            (fx / f"mediapipe_selfie_{variant}.ref.json").write_text(json.dumps({
                "variant": variant, "h": src["h"], "w": src["w"],
                "input": np.round(img.ravel(), 7).tolist(), "expected": np.round(ref.ravel(), 7).tolist()}))

        manifest["files"][src["out"]] = {"variant": variant, "bytes": dst.stat().st_size,
                                         "sha256": sha256(dst), "source": src["url"],
                                         "source_sha256": src["sha256"]}

    for f in ("LICENSE", "NOTICE.md"):
        shutil.copy(ROOT / "third_party/mediapipe" / f, out / f)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
