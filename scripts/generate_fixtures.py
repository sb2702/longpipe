"""
Generate op-level test fixtures for the Longpipe SDK.

Outputs JSON files to sdk/tests/fixtures/ containing small deterministic
inputs and PyTorch-computed expected outputs for each op type.

Usage:
    python sdk/scripts/generate_fixtures.py
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def to_nhwc(t):
    """[B, C, H, W] → flat float32 list in NHWC vec4 layout."""
    nhwc = t.permute(0, 2, 3, 1).contiguous()
    B, H, W, C = nhwc.shape
    assert C % 4 == 0
    return nhwc.reshape(B, H, W, C // 4, 4).flatten().cpu().numpy().tolist()


def conv_weights(w):
    """[O, I, KH, KW] → mat4x4 list for conv2d.wgsl."""
    O, I, KH, KW = w.shape
    assert O % 4 == 0 and I % 4 == 0
    out = []
    for ky in range(KH):
        for kx in range(KW):
            for o in range(O // 4):
                for i in range(I // 4):
                    mat = w[o*4:(o+1)*4, i*4:(i+1)*4, ky, kx].cpu().numpy()
                    out.extend(mat.T.flatten().tolist())
    return out


# ── Standalone ops ────────────────────────────────────────────────────────────

def generate_sigmoid():
    C, H, W = 16, 8, 8
    x = torch.randn(1, C, H, W) * 4  # wide range exercises saturation
    y = torch.sigmoid(x)
    return {
        "channels": C,
        "input_shape": [1, C, H, W],
        "input": to_nhwc(x),
        "expected_output": to_nhwc(y),
    }


def generate_bilinear_upsample():
    C, H, W = 16, 8, 8
    x = torch.randn(1, C, H, W)
    y = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    out_H, out_W = y.shape[2], y.shape[3]
    return {
        "channels": C,
        "input_shape":  [1, C, H, W],
        "output_shape": [1, C, out_H, out_W],
        "input": to_nhwc(x),
        "expected_output": to_nhwc(y),
    }


def generate_channel_concat():
    C, H, W = 16, 8, 8
    a = torch.randn(1, C, H, W)
    b = torch.randn(1, C, H, W)
    y = torch.cat([a, b], dim=1)
    return {
        "a_channels": C,
        "b_channels": C,
        "out_channels": C * 2,
        "height": H,
        "width": W,
        "input_shape":  [1, C, H, W],
        "output_shape": [1, C * 2, H, W],
        "input_a": to_nhwc(a),
        "input_b": to_nhwc(b),
        "expected_output": to_nhwc(y),
    }


# ── Fused ops ─────────────────────────────────────────────────────────────────

def generate_conv2d_add():
    C, H, W = 8, 8, 8
    conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
    x    = torch.randn(1, C, H, W)
    skip = torch.randn(1, C, H, W)
    with torch.no_grad():
        y = conv(x) + skip
    return {
        "in_channels":  C,
        "out_channels": C,
        "kernel_size":  3,
        "stride":       1,
        "padding":      1,
        "input_shape":  [1, C, H, W],
        "input":   to_nhwc(x),
        "skip":    to_nhwc(skip),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_upsample_concat():
    Ca, Cb, H, W = 16, 16, 8, 8
    a = torch.randn(1, Ca, H, W)
    b = torch.randn(1, Cb, H * 2, W * 2)
    up = F.interpolate(a, scale_factor=2, mode="bilinear", align_corners=False)
    y  = torch.cat([up, b], dim=1)
    return {
        "a_channels":   Ca,
        "b_channels":   Cb,
        "out_channels": Ca + Cb,
        "in_h": H, "in_w": W,
        "out_h": H * 2, "out_w": W * 2,
        "a_shape": [1, Ca, H,     W],
        "b_shape": [1, Cb, H * 2, W * 2],
        "output_shape": [1, Ca + Cb, H * 2, W * 2],
        "input_a": to_nhwc(a),
        "input_b": to_nhwc(b),
        "expected_output": to_nhwc(y),
    }


def generate_upsample_conv1x1():
    C_in, C_out, H, W = 16, 8, 8, 8
    conv = nn.Conv2d(C_in, C_out, kernel_size=1, bias=True)
    x    = torch.randn(1, C_in, H, W)
    with torch.no_grad():
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        y  = conv(up)
    return {
        "in_channels":  C_in,
        "out_channels": C_out,
        "in_h": H, "in_w": W,
        "out_h": H * 2, "out_w": W * 2,
        "input_shape":  [1, C_in, H, W],
        "output_shape": [1, C_out, H * 2, W * 2],
        "input":   to_nhwc(x),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_upsample_sigmoid():
    C, H, W = 16, 8, 8
    x = torch.randn(1, C, H, W)
    with torch.no_grad():
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        y  = torch.sigmoid(up)
    return {
        "channels": C,
        "in_h": H, "in_w": W,
        "out_h": H * 2, "out_w": W * 2,
        "input_shape":  [1, C, H, W],
        "output_shape": [1, C, H * 2, W * 2],
        "input": to_nhwc(x),
        "expected_output": to_nhwc(y),
    }


# ── Registry ──────────────────────────────────────────────────────────────────

FIXTURES = {
    "sigmoid":              generate_sigmoid,
    "bilinear_upsample_2x": generate_bilinear_upsample,
    "channel_concat":       generate_channel_concat,
    "conv2d_add":           generate_conv2d_add,
    "upsample_concat":      generate_upsample_concat,
    "upsample_conv1x1":     generate_upsample_conv1x1,
    "upsample_sigmoid":     generate_upsample_sigmoid,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        data = fn()
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"✓ {path}")
