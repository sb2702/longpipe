#!/usr/bin/env python3
"""Longpipe vs MediaPipe selfie segmentation on the P3M-10k validation sets.

Evaluates, on P3M-500-P (face-blurred) and P3M-500-NP (not blurred):

  longpipe-<tier>   the five official Longpipe ONNX matting models (xs, small,
                    medium, large, xl) from huggingface.co/Longpipe/longpipe,
                    at a pinned revision
  mediapipe-square  MediaPipe Tasks ImageSegmenter + selfie_segmenter.tflite
                    (256x256, the default model in MediaPipe's web docs)
  mediapipe-landscape  ... + selfie_segmenter_landscape.tflite (144x256)

Every prediction is scored at the ground-truth resolution with the metric code
of the P3M paper (github.com/JizhiziLi/P3M, core/evaluate.py), plus IoU at 0.5.
See ../METHODOLOGY.md for the protocol and its caveats.

Dataset: P3M-10k is distributed by its authors under a release agreement and
is NOT included here. See ../README.md ("Getting P3M-10k") for how to get it.

Usage:
    python eval_p3m.py --p3m-root /path/to/P3M-10k
    python eval_p3m.py --p3m-root /path/to/P3M-10k --limit 20          # quick check
    python eval_p3m.py --p3m-root /path/to/P3M-10k --protocol legacy-readme

Protocols:
    standard       (default) every image of both sets, full frame, all metrics
    legacy-readme  reproduces the protocol of the pre-2026-11 README chart:
                   landscape images only (w/h > 1.05 -> 499 of 1000), centre-
                   cropped to 16:9, MAD + IoU only, MediaPipe landscape model
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp_proc
import os
import platform
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATASET_HELP = (
    'P3M-10k is not bundled with this repo. Download it from the authors '
    '(https://github.com/JizhiziLi/P3M, "PPT Setting and P3M-10k Dataset"), accept the '
    'release agreement, unzip it, and pass the extracted folder (the one containing '
    '"validation/") as --p3m-root. Full instructions: benchmarks/README.md.'
)

# ── Models ───────────────────────────────────────────────────────────────────

HF_REPO = 'Longpipe/longpipe'
HF_REVISION = '521aefefe13e238aeb8bab0916e6dfe2456bbdd1'   # pinned; override with --hf-revision

# Canvas (= network input = alpha output) size per tier, landscape (W, H).
# Mirrors src/model/tier_config.ts TIER_CONFIG[*].canvasRes. Portrait = swapped.
LONGPIPE_TIERS = {
    'xs':     (384, 240),
    'small':  (384, 224),
    'medium': (512, 320),
    'large':  (640, 400),
    'xl':     (1280, 768),
}

MP_BASE = 'https://storage.googleapis.com/mediapipe-models/image_segmenter'
MEDIAPIPE_MODELS = {
    # name: (url, sha256, native output (W, H))
    'square': (f'{MP_BASE}/selfie_segmenter/float16/1/selfie_segmenter.tflite',
               '191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b', (256, 256)),
    'landscape': (f'{MP_BASE}/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite',
                  '490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e', (256, 144)),
}

ALL_METHODS = [f'longpipe-{t}' for t in LONGPIPE_TIERS] + [f'mediapipe-{m}' for m in MEDIAPIPE_MODELS]

SETS = {
    # set name: (folder under validation/, image subfolder)
    'P3M-500-P':  ('P3M-500-P', 'blurred_image'),
    'P3M-500-NP': ('P3M-500-NP', 'original_image'),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_mediapipe_model(name: str, cache: Path) -> Path:
    url, digest, _ = MEDIAPIPE_MODELS[name]
    path = cache / 'mediapipe' / url.rsplit('/', 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'  downloading {url}')
        urllib.request.urlretrieve(url, path)
    got = sha256(path)
    if got != digest:
        raise SystemExit(f'{path}: sha256 {got} != expected {digest}')
    return path


def fetch_longpipe_model(tier: str, orientation: str, cache: Path, revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, f'onnx/{tier}/matting_{orientation}.onnx',
                                revision=revision, cache_dir=str(cache / 'hf')))


# ── Resampling ───────────────────────────────────────────────────────────────

def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize, half-pixel centres, no antialiasing — the same sampling as
    torch F.interpolate(mode='bilinear', align_corners=False), cv2 INTER_LINEAR and
    a GPU bilinear texture fetch (what the Longpipe SDK's Input op and alpha
    upscaler do). img: (H, W) or (H, W, C) float32."""
    in_h, in_w = img.shape[:2]

    def axis(n_in, n_out):
        x = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
        x = np.clip(x, 0, n_in - 1)
        i0 = np.floor(x).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (x - i0).astype(np.float32)

    y0, y1, wy = axis(in_h, out_h)
    x0, x1, wx = axis(in_w, out_w)
    # separable: rows first, then columns
    wy_ = wy.reshape((-1,) + (1,) * (img.ndim - 1))
    rows = img[y0] * (1 - wy_) + img[y1] * wy_
    wx_ = wx.reshape((1, -1) + (1,) * (img.ndim - 2))
    return (rows[:, x0] * (1 - wx_) + rows[:, x1] * wx_).astype(np.float32)


def center_crop_to_aspect(arrs, target_w, target_h):
    """Centre crop every array to target_w/target_h (legacy-readme protocol; same
    arithmetic as the old training/benchmark_accuracy.py)."""
    h, w = arrs[0].shape[:2]
    target, r = target_w / target_h, w / h
    if r > target:
        nw = int(h * target); x0 = (w - nw) // 2
        return [a[:, x0:x0 + nw] for a in arrs]
    if r < target:
        nh = int(w / target); y0 = (h - nh) // 2
        return [a[y0:y0 + nh, :] for a in arrs]
    return arrs


# ── Metrics (P3M core/evaluate.py, alpha in [0, 1]) ─────────────────────────

def metrics_whole(pred, gt):
    n = pred.size
    diff = pred - gt
    return {'sad': float(np.abs(diff).sum() / 1000), 'mse': float((diff ** 2).sum() / n),
            'mad': float(np.abs(diff).sum() / n)}


def metrics_trimap(pred, gt, trimap):
    """P3M calculate_sad_mse_mad: known regions are set from the trimap, so error is
    counted only in the unknown band; MSE/MAD are normalised by its pixel count."""
    p = pred.copy()
    p[trimap == 255] = 1.0
    p[trimap == 0] = 0.0
    unknown = float((trimap == 128).sum())
    if unknown == 0:
        unknown = trimap.size - float((trimap == 255).sum()) - float((trimap == 0).sum())
    diff = p - gt
    return {'sad_unknown': float(np.abs(diff).sum() / 1000),
            'mse_unknown': float((diff ** 2).sum() / unknown) if unknown else 0.0,
            'mad_unknown': float(np.abs(diff).sum() / unknown) if unknown else 0.0}


def metric_grad(pred, gt):
    from scipy.ndimage import gaussian_filter
    pd_x = gaussian_filter(pred, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pred, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x ** 2 + pd_y ** 2)
    gt_mag = np.sqrt(gt_x ** 2 + gt_y ** 2)
    return float(np.square(pd_mag - gt_mag).sum() / 10)


def metric_conn(pred, gt, step=0.1):
    """P3M compute_connectivity_loss_whole_image, with scipy.ndimage.label (4-connectivity,
    same as skimage connectivity=1) instead of skimage. The unused distance transform of
    the reference code is omitted; the result is identical."""
    from scipy.ndimage import label
    pred, gt = pred.astype(np.float64), gt.astype(np.float64)   # match the reference's dtype
    h, w = pred.shape
    thresh = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh.size):
        both = (pred >= thresh[i]) & (gt >= thresh[i])
        lab, n = label(both)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        omega = lab == int(np.argmax(sizes))
        flag = (l_map == -1) & ~omega
        l_map[flag] = thresh[i - 1]
    l_map[l_map == -1] = 1
    d_pd = pred - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    return float(np.abs(phi_pd - phi_gt).sum() / 1000)


def metric_iou(pred, gt):
    pb, gb = pred >= 0.5, gt >= 0.5
    union = np.logical_or(pb, gb).sum()
    return float(np.logical_and(pb, gb).sum() / union) if union else 1.0


# ── Methods ──────────────────────────────────────────────────────────────────

class Runner:
    """Holds one inference session per method; created once per worker process."""

    def __init__(self, methods, model_paths):
        self.methods = methods
        self.lp = {}
        self.mp = {}
        for m in methods:
            if m.startswith('longpipe-'):
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                tier = m.split('-', 1)[1]
                self.lp[tier] = {o: ort.InferenceSession(str(model_paths[m][o]), so,
                                                         providers=['CPUExecutionProvider'])
                                 for o in ('landscape', 'portrait')}
            else:
                from mediapipe.tasks.python import BaseOptions, vision
                opts = vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=str(model_paths[m])),
                    running_mode=vision.RunningMode.IMAGE,
                    output_confidence_masks=True, output_category_mask=False)
                self.mp[m.split('-', 1)[1]] = vision.ImageSegmenter.create_from_options(opts)

    def predict(self, method, rgb_u8, legacy=False):
        """rgb_u8: (H, W, 3) uint8. Returns (alpha (H, W) float32 in [0, 1], native mask (W, H))."""
        h, w = rgb_u8.shape[:2]
        if method.startswith('longpipe-'):
            tier = method.split('-', 1)[1]
            cw, ch = LONGPIPE_TIERS[tier]
            orient = 'landscape' if (legacy or w >= h) else 'portrait'
            if orient == 'portrait':
                cw, ch = ch, cw
            x = resize_bilinear(rgb_u8.astype(np.float32) / 255.0, ch, cw)
            x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
            alpha = self.lp[tier][orient].run(['alpha'], {'rgb': x})[0][0, 0]
            alpha = np.clip(alpha, 0, 1).astype(np.float32)
            return resize_bilinear(alpha, h, w), (cw, ch)
        import mediapipe as mp
        name = method.split('-', 1)[1]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8))
        res = self.mp[name].segment(img)
        conf = res.confidence_masks[-1].numpy_view()
        conf = np.asarray(conf, dtype=np.float32).reshape(conf.shape[0], conf.shape[1])
        if conf.shape != (h, w):   # MediaPipe returns masks at input size; defensive
            conf = resize_bilinear(conf, h, w)
        return np.clip(conf, 0, 1), MEDIAPIPE_MODELS[name][2]


_RUNNER = None
_CFG = None


def _init(methods, model_paths, cfg):
    global _RUNNER, _CFG
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    _RUNNER = Runner(methods, model_paths)
    _CFG = cfg


def load_sample(s):
    rgb = np.array(Image.open(s['image']).convert('RGB'))
    gt = np.array(Image.open(s['mask']))
    # float64, exactly as P3M's test.py loads it (np.array(...) / 255.): the
    # connectivity metric's thresholds (0.1, 0.2, ...) land exactly on some k/255 values,
    # so float32 alpha would shift a few pixels across a threshold.
    gt = (gt[:, :, 0] if gt.ndim > 2 else gt) / 255.0
    tri = np.array(Image.open(s['trimap']))
    tri = tri[:, :, 0] if tri.ndim > 2 else tri
    if gt.shape != rgb.shape[:2] or tri.shape != rgb.shape[:2]:
        raise ValueError(f'{s["name"]}: image/mask/trimap size mismatch')
    return rgb, gt, tri


def eval_one(s):
    legacy = _CFG['protocol'] == 'legacy-readme'
    rgb, gt, tri = load_sample(s)
    if legacy:
        rgb, gt, tri = center_crop_to_aspect([rgb, gt, tri], 16, 9)
    rows = []
    for m in _RUNNER.methods:
        t = time.perf_counter()
        pred, native = _RUNNER.predict(m, rgb, legacy=legacy)
        infer_s = time.perf_counter() - t
        row = {'method': m, 'set': s['set'], 'image': s['name'], 'orientation': s['orientation'],
               'width': rgb.shape[1], 'height': rgb.shape[0], 'mask_w': native[0], 'mask_h': native[1]}
        row.update(metrics_whole(pred, gt))
        row['iou'] = metric_iou(pred, gt)
        if not legacy:
            row.update(metrics_trimap(pred, gt, tri))
            row['grad'] = metric_grad(pred, gt)
            row['conn'] = metric_conn(pred, gt)
        row['infer_s'] = infer_s
        rows.append(row)
    return rows


# ── Dataset discovery ────────────────────────────────────────────────────────

def discover(root: Path, sets, limit):
    if not root.is_dir():
        raise SystemExit(f'--p3m-root {root} does not exist.\n{DATASET_HELP}')
    if not (root / 'validation').is_dir() and (root / 'P3M-10k' / 'validation').is_dir():
        root = root / 'P3M-10k'   # pointed at the folder the zip was extracted into
    samples, problems = [], []
    for name in sets:
        folder, img_sub = SETS[name]
        base = root / 'validation' / folder
        for sub in (img_sub, 'mask', 'trimap'):
            if not (base / sub).is_dir():
                problems.append(str(base / sub))
        if problems:
            continue
        imgs = sorted(p for p in (base / img_sub).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not imgs:
            problems.append(f'{base / img_sub} (no images)')
            continue
        if limit:
            imgs = imgs[:limit]
        for p in imgs:
            mask, tri = base / 'mask' / f'{p.stem}.png', base / 'trimap' / f'{p.stem}.png'
            if not mask.exists() or not tri.exists():
                problems.append(f'{mask} / {tri}')
                continue
            with Image.open(p) as im:
                w, h = im.size
            r = w / h
            orient = 'landscape' if r > 1.05 else ('portrait' if r < 0.95 else 'square')
            samples.append({'set': name, 'name': p.stem, 'image': str(p), 'mask': str(mask),
                            'trimap': str(tri), 'orientation': orient})
    if problems:
        shown = '\n  '.join(problems[:10])
        raise SystemExit(
            f'Expected P3M-10k layout not found under {root}. Missing:\n  {shown}\n\n'
            'Expected (the layout of the official download, unchanged):\n'
            '  P3M-10k/validation/P3M-500-P/{blurred_image,mask,trimap}/\n'
            '  P3M-10k/validation/P3M-500-NP/{original_image,mask,trimap}/\n\n' + DATASET_HELP)
    return samples


# ── Aggregation / reporting ──────────────────────────────────────────────────

METRIC_KEYS = ['sad', 'mse', 'mad', 'grad', 'conn', 'iou', 'sad_unknown', 'mse_unknown', 'mad_unknown']


def aggregate(rows, methods):
    groups = {}
    for r in rows:
        keys = [('all', 'all'), (r['set'], 'all')]
        keys += [('all', r['orientation']), (r['set'], r['orientation'])]
        for k in keys:
            groups.setdefault((r['method'],) + k, []).append(r)
    out = {}
    for (m, s, o), rs in sorted(groups.items()):
        out.setdefault(m, {}).setdefault(s, {})[o] = {
            'n': len(rs),
            **{k: float(np.mean([r[k] for r in rs])) for k in METRIC_KEYS if k in rs[0]},
            'mask_res': sorted({f'{r["mask_w"]}x{r["mask_h"]}' for r in rs}),
            'cpu_ms_per_image': 1000 * float(np.median([r['infer_s'] for r in rs])),
        }
    return out


def fmt_table(agg, methods, set_name, orient, cols):
    head = '| Method | n | ' + ' | '.join(c[1] for c in cols) + ' |'
    sep = '|---|---:|' + '---:|' * len(cols)
    lines = [head, sep]
    for m in methods:
        a = agg.get(m, {}).get(set_name, {}).get(orient)
        if not a:
            continue
        cells = []
        for key, _, f in cols:
            cells.append(f.format(a[key]) if key in a else '–')
        lines.append(f'| {m} | {a["n"]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


STD_COLS = [('mad', 'MAD', '{:.4f}'), ('mse', 'MSE', '{:.4f}'), ('sad', 'SAD', '{:.2f}'),
            ('grad', 'Grad', '{:.2f}'), ('conn', 'Conn', '{:.2f}'), ('iou', 'IoU@0.5', '{:.4f}'),
            ('mad_unknown', 'MAD (trimap unknown)', '{:.4f}'), ('sad_unknown', 'SAD (trimap unknown)', '{:.2f}')]
LEGACY_COLS = [('mad', 'MAD', '{:.4f}'), ('iou', 'IoU@0.5', '{:.4f}')]


def environment(args, model_paths):
    import importlib.metadata as md
    vers = {}
    for p in ('numpy', 'pillow', 'scipy', 'onnxruntime', 'mediapipe', 'huggingface_hub'):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    cpu = platform.processor()
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip(); break
    except OSError:
        pass
    models = {}
    for m, p in model_paths.items():
        if isinstance(p, dict):
            models[m] = {o: {'file': f'onnx/{m.split("-", 1)[1]}/matting_{o}.onnx', 'sha256': sha256(Path(q))}
                         for o, q in p.items()}
        else:
            name = m.split('-', 1)[1]
            models[m] = {'url': MEDIAPIPE_MODELS[name][0], 'sha256': sha256(Path(p))}
    return {'date': date.today().isoformat(), 'python': sys.version.split()[0], 'packages': vers,
            'platform': platform.platform(), 'cpu': cpu, 'workers': args.workers,
            'hf_repo': HF_REPO, 'hf_revision': args.hf_revision, 'models': models,
            'onnxruntime_provider': 'CPUExecutionProvider (fp32, 1 thread per worker)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p3m-root', required=True, type=Path,
                    help='extracted P3M-10k folder (contains validation/P3M-500-P and validation/P3M-500-NP)')
    ap.add_argument('--protocol', choices=['standard', 'legacy-readme'], default='standard')
    ap.add_argument('--methods', nargs='+', default=None, choices=ALL_METHODS)
    ap.add_argument('--sets', nargs='+', default=list(SETS), choices=list(SETS))
    ap.add_argument('--limit', type=int, default=0, help='first N images per set (0 = all)')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--cache-dir', type=Path, default=HERE / '.cache')
    ap.add_argument('--hf-revision', default=HF_REVISION)
    ap.add_argument('--out-dir', type=Path, default=HERE / 'results')
    ap.add_argument('--tag', default=None, help='results file stem (default <date>-p3m[-legacy])')
    args = ap.parse_args()

    legacy = args.protocol == 'legacy-readme'
    methods = args.methods or (ALL_METHODS if not legacy else
                               [f'longpipe-{t}' for t in LONGPIPE_TIERS] + ['mediapipe-landscape'])

    samples = discover(args.p3m_root.expanduser().resolve(), args.sets, args.limit)
    if legacy:
        samples = [s for s in samples if s['orientation'] == 'landscape']
    for name in args.sets:
        n = sum(s['set'] == name for s in samples)
        by = {o: sum(s['set'] == name and s['orientation'] == o for s in samples)
              for o in ('landscape', 'portrait', 'square')}
        print(f'{name}: {n} images ({by["landscape"]} landscape, {by["portrait"]} portrait, {by["square"]} square)')
    if not samples:
        raise SystemExit('No images to evaluate.')

    print('Fetching models…')
    model_paths = {}
    for m in methods:
        kind, name = m.split('-', 1)
        if kind == 'longpipe':
            model_paths[m] = {o: fetch_longpipe_model(name, o, args.cache_dir, args.hf_revision)
                              for o in ('landscape', 'portrait')}
        else:
            model_paths[m] = fetch_mediapipe_model(name, args.cache_dir)

    print(f'Evaluating {len(methods)} methods on {len(samples)} images, protocol={args.protocol}, '
          f'{args.workers} workers…')
    t0 = time.time()
    rows = []
    cfg = {'protocol': args.protocol}
    ctx = mp_proc.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init, initargs=(methods, model_paths, cfg)) as pool:
        for i, rs in enumerate(pool.imap(eval_one, samples, chunksize=1)):
            rows.extend(rs)
            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                print(f'  {i + 1}/{len(samples)}  ({time.time() - t0:.0f}s)', flush=True)

    agg = aggregate(rows, methods)
    env = environment(args, model_paths)
    env['wall_s'] = round(time.time() - t0, 1)

    stem = args.tag or f'{env["date"]}-p3m' + ('-legacy' if legacy else '')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {'protocol': args.protocol, 'limit': args.limit, 'methods': methods,
              'environment': env, 'results': agg}
    (args.out_dir / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    with gzip.open(args.out_dir / f'{stem}-per-image.csv.gz', 'wt', newline='') as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['method'], r['set'], r['image'])):
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})

    cols = LEGACY_COLS if legacy else STD_COLS
    md = [f'# P3M-10k validation: Longpipe vs MediaPipe ({args.protocol} protocol)', '',
          f'Generated by `benchmarks/quality/eval_p3m.py --protocol {args.protocol}` on {env["date"]}. '
          'Methodology: [`../../METHODOLOGY.md`](../../METHODOLOGY.md).', '']
    if args.limit:
        md += [f'**Partial run: first {args.limit} images per set only.**', '']
    md += ['## Environment', '', '```', json.dumps({k: v for k, v in env.items() if k != 'models'}, indent=2), '```', '']
    md += ['## Native mask resolution (before upsampling to ground-truth size)', '',
           '| Method | Model output (W×H) |', '|---|---|']
    for m in methods:
        res = sorted({r for s in agg[m].values() for o in s.values() for r in o['mask_res']})
        md.append(f'| {m} | {", ".join(res)} |')
    md.append('')
    scopes = [('all', 'all', 'Both sets (P3M-500-P + P3M-500-NP)')] + [(s, 'all', s) for s in args.sets]
    scopes += [('all', 'landscape', 'Landscape images only (w/h > 1.05), both sets'),
               ('all', 'portrait', 'Portrait images only (w/h < 0.95), both sets')]
    for s, o, title in scopes:
        if any(agg[m].get(s, {}).get(o) for m in methods):
            md += [f'## {title}', '', fmt_table(agg, methods, s, o, cols), '']
    md += ['Lower is better for every column except IoU. SAD and Conn are /1000, Grad is /10, as in '
           "P3M's evaluate.py. Per-image values: "
           f'`{stem}-per-image.csv.gz`. Model files and hashes: `{stem}.json`.', '']
    (args.out_dir / f'{stem}.md').write_text('\n'.join(md))

    print()
    for s, o, title in scopes:
        if any(agg[m].get(s, {}).get(o) for m in methods):
            print(f'{title}\n{fmt_table(agg, methods, s, o, cols)}\n')
    print(f'Wrote {args.out_dir / stem}.{{json,md}} and -per-image.csv.gz  ({env["wall_s"]}s)')


if __name__ == '__main__':
    main()
