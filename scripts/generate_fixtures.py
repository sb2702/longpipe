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


def generate_elementwise_add():
    C, H, W = 16, 8, 8
    a = torch.randn(1, C, H, W)
    b = torch.randn(1, C, H, W)
    y = a + b
    return {
        "channels": C,
        "input_shape": [1, C, H, W],
        "input1": to_nhwc(a),
        "input2": to_nhwc(b),
        "expected_output": to_nhwc(y),
    }


def generate_conv2d_1x1():
    in_C, out_C, H, W = 8, 16, 8, 8
    conv = nn.Conv2d(in_C, out_C, kernel_size=1, bias=True)
    x = torch.randn(1, in_C, H, W)
    with torch.no_grad():
        y = conv(x)
    return {
        "kernel_size":  1, "stride": 1, "padding": 0,
        "in_channels":  in_C, "out_channels": out_C,
        "input_shape":  [1, in_C,  H, W],
        "output_shape": [1, out_C, H, W],
        "input":   to_nhwc(x),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_conv2d_3x3():
    in_C, out_C, H, W = 8, 16, 8, 8
    conv = nn.Conv2d(in_C, out_C, kernel_size=3, padding=1, bias=True)
    x = torch.randn(1, in_C, H, W)
    with torch.no_grad():
        y = conv(x)
    return {
        "kernel_size":  3, "stride": 1, "padding": 1,
        "in_channels":  in_C, "out_channels": out_C,
        "input_shape":  [1, in_C,  H, W],
        "output_shape": [1, out_C, H, W],
        "input":   to_nhwc(x),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_conv2d_3x3_stride2():
    in_C, out_C, H, W = 8, 16, 16, 16
    conv = nn.Conv2d(in_C, out_C, kernel_size=3, stride=2, padding=1, bias=True)
    x = torch.randn(1, in_C, H, W)
    with torch.no_grad():
        y = conv(x)
    return {
        "kernel_size":  3, "stride": 2, "padding": 1,
        "in_channels":  in_C, "out_channels": out_C,
        "input_shape":  [1, in_C,  H, W],
        "output_shape": list(y.shape),
        "input":   to_nhwc(x),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_conv2d_3x3_bn_fused():
    # Verifies pre-fusing BN into conv weights produces the same output as
    # running conv→BN explicitly.
    in_C, out_C, H, W = 8, 16, 8, 8
    conv = nn.Conv2d(in_C, out_C, kernel_size=3, padding=1, bias=True)
    bn   = nn.BatchNorm2d(out_C, eps=0.001).eval()
    bn.running_mean = torch.randn(out_C)
    bn.running_var  = torch.rand(out_C).abs() + 0.1
    bn.weight.data  = torch.randn(out_C)
    bn.bias.data    = torch.randn(out_C)

    x = torch.randn(1, in_C, H, W)
    with torch.no_grad():
        y_separate = bn(conv(x))

    std   = torch.sqrt(bn.running_var + 0.001)
    scale = bn.weight / std
    fused_w = conv.weight * scale.view(-1, 1, 1, 1)
    fused_b = (conv.bias - bn.running_mean) * scale + bn.bias

    return {
        "kernel_size":  3, "stride": 1, "padding": 1,
        "in_channels":  in_C, "out_channels": out_C,
        "input_shape":  [1, in_C,  H, W],
        "output_shape": [1, out_C, H, W],
        "input":   to_nhwc(x),
        "weights": conv_weights(fused_w.detach()),
        "bias":    fused_b.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y_separate),
    }


def generate_depthwise_3x3():
    C, H, W = 16, 8, 8
    conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True)
    x = torch.randn(1, C, H, W)
    with torch.no_grad():
        y = conv(x)
    return {
        "kernel_size": 3, "stride": 1, "padding": 1, "channels": C,
        "input_shape":  [1, C, H, W],
        "output_shape": [1, C, H, W],
        "input":   to_nhwc(x),
        "weights": dw_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def _generate_mbconv(stride: int, has_residual: bool):
    in_C, mid_C, out_C = 16, 32, 16 if has_residual else 32
    H, W = (8, 8) if stride == 1 else (16, 16)
    expand = nn.Conv2d(in_C,  mid_C, kernel_size=1, bias=True)
    dw     = nn.Conv2d(mid_C, mid_C, kernel_size=3, stride=stride, padding=1, groups=mid_C, bias=True)
    proj   = nn.Conv2d(mid_C, out_C, kernel_size=1, bias=True)

    x = torch.randn(1, in_C, H, W)
    with torch.no_grad():
        a = torch.clamp(expand(x), 0, 6)
        b = torch.clamp(dw(a),     0, 6)
        c = proj(b)
        y = (x + c) if has_residual else c

    out_H, out_W = c.shape[2], c.shape[3]
    return {
        "kernel_size": 3, "stride": stride, "padding": 1,
        "in_channels": in_C, "mid_channels": mid_C, "out_channels": out_C,
        "input_shape":  [1, in_C,  H, W],
        "output_shape": [1, out_C, out_H, out_W],
        "input":          to_nhwc(x),
        "expand_weights": conv_weights(expand.weight.detach()),
        "expand_bias":    expand.bias.detach().cpu().numpy().tolist(),
        "dw_weights":     dw_weights(dw.weight.detach()),
        "dw_bias":        dw.bias.detach().cpu().numpy().tolist(),
        "proj_weights":   conv_weights(proj.weight.detach()),
        "proj_bias":      proj.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def generate_mbconv_k3_s1_residual():
    return _generate_mbconv(stride=1, has_residual=True)


def generate_mbconv_k3_s2():
    return _generate_mbconv(stride=2, has_residual=False)


def generate_tanh():
    C, H, W = 16, 8, 8
    x = torch.randn(1, C, H, W) * 2.0   # cover saturation regions
    y = torch.tanh(x)
    return {
        "channels": C,
        "input_shape": [1, C, H, W],
        "input": to_nhwc(x),
        "expected_output": to_nhwc(y),
    }


def generate_elementwise_mul():
    # Mirrors elementwise_add: two equal-shape tensors, output = a * b.
    C, H, W = 16, 8, 8
    a = torch.randn(1, C, H, W)
    b = torch.randn(1, C, H, W)
    y = a * b
    return {
        "channels": C,
        "input_shape": [1, C, H, W],
        "input1": to_nhwc(a),
        "input2": to_nhwc(b),
        "expected_output": to_nhwc(y),
    }


def generate_gru_update():
    # Fused (1 - z) * h_prev + z * h_til, used inside ConvGRU.
    # z is the output of a sigmoid so live in (0, 1) in practice.
    C, H, W = 16, 8, 8
    z      = torch.sigmoid(torch.randn(1, C, H, W))
    h_prev = torch.randn(1, C, H, W)
    h_til  = torch.tanh(torch.randn(1, C, H, W))
    h_new  = (1.0 - z) * h_prev + z * h_til
    return {
        "channels": C,
        "input_shape": [1, C, H, W],
        "z":      to_nhwc(z),
        "h_prev": to_nhwc(h_prev),
        "h_til":  to_nhwc(h_til),
        "expected_output": to_nhwc(h_new),
    }


def generate_gamma_residual():
    # Per-channel scaled residual b + γ * h_new, used at the end of ConvGRU.
    # γ is one f32 per channel — laid out as a flat array of length C; the
    # shader / runtime treats it as c_groups vec4s.
    C, H, W = 16, 8, 8
    b      = torch.randn(1, C, H, W)
    h_new  = torch.randn(1, C, H, W)
    gamma  = torch.randn(C)
    out    = b + gamma.view(1, -1, 1, 1) * h_new
    return {
        "channels": C,
        "height": H,
        "width":  W,
        "input_shape": [1, C, H, W],
        "b":     to_nhwc(b),
        "h_new": to_nhwc(h_new),
        "gamma": gamma.cpu().numpy().tolist(),
        "expected_output": to_nhwc(out),
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


# ── Blocks ───────────────────────────────────────────────────────────────────

def generate_depthwise_separable():
    """Stage-0-sized: 32→16, k=3, s=1. DW has relu6, PW has no activation."""
    C_in, C_out, H, W = 32, 16, 8, 8
    dw = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, groups=C_in, bias=True)
    pw = nn.Conv2d(C_in, C_out, kernel_size=1, bias=True)
    x  = torch.randn(1, C_in, H, W)
    with torch.no_grad():
        after_dw = torch.clamp(dw(x), 0, 6)
        y        = pw(after_dw)
    return {
        "in_channels":  C_in,
        "out_channels": C_out,
        "kernel_size":  3,
        "stride":       1,
        "padding":      1,
        "input_shape":  [1, C_in, H, W],
        "output_shape": [1, C_out, H, W],
        "input":    to_nhwc(x),
        "dw":       { "weights": dw_weights(dw.weight.detach()),
                      "bias":    dw.bias.detach().cpu().numpy().tolist() },
        "pw":       { "weights": conv_weights(pw.weight.detach()),
                      "bias":    pw.bias.detach().cpu().numpy().tolist() },
        "expected_output": to_nhwc(y),
    }


def generate_convgru_block():
    """One-timestep ConvGRU forward — mirrors training/models/temporal_model.py:ConvGRU.

    Pre-splits inputs into (a, b) and the gates conv into (z_conv, r_conv) so
    the SDK block reuses existing primitives without needing channel-slice or
    split-and-activate shaders.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from models.temporal_model import ConvGRU

    # Channel layout: C = 16, split 50/50 → passthrough = recurrent = 8.
    # 2c = 16 for gates conv. All multiples of 4.
    C, H, W = 16, 8, 8
    gru = ConvGRU(channels=C, kernel=3, split_ratio=0.5).eval()
    # Cand starts zero-init (identity-at-init); fill with random so the path
    # is actually exercised.
    nn.init.kaiming_normal_(gru.cand.weight)
    nn.init.zeros_(gru.cand.bias)
    with torch.no_grad():
        gru.gamma.copy_(torch.randn_like(gru.gamma))

    passthrough = gru.passthrough_ch
    recurrent   = gru.recurrent_ch

    x      = torch.randn(1, C, H, W)
    h_prev = torch.randn(1, recurrent, H, W)
    a = x[:, :passthrough]
    b = x[:, passthrough:]

    # Capture all intermediates so a debug test can pinpoint divergence.
    with torch.no_grad():
        cat_bh   = torch.cat([b, h_prev], dim=1)
        z_pre    = gru.gates(cat_bh)[:, :recurrent]
        r_pre    = gru.gates(cat_bh)[:, recurrent:]
        z        = torch.sigmoid(z_pre)
        r        = torch.sigmoid(r_pre)
        rh       = r * h_prev
        cat_brh  = torch.cat([b, rh], dim=1)
        cand_pre = gru.cand(cat_brh)
        h_til    = torch.tanh(cand_pre)
        h_new    = (1.0 - z) * h_prev + z * h_til
        b_out    = b + gru.gamma.view(1, -1, 1, 1) * h_new
        out      = torch.cat([a, b_out], dim=1)

    # Split gates conv (out=2c) into z_conv (out=c) and r_conv (out=c).
    gates_w = gru.gates.weight.detach()
    gates_b = gru.gates.bias.detach()
    z_w, r_w = gates_w[:recurrent], gates_w[recurrent:]
    z_b, r_b = gates_b[:recurrent], gates_b[recurrent:]

    return {
        "channels":    C,
        "passthrough": passthrough,
        "recurrent":   recurrent,
        "height":      H,
        "width":       W,
        "input_shape": [1, C, H, W],
        "a":      to_nhwc(a),
        "b":      to_nhwc(b),
        "h_prev": to_nhwc(h_prev),
        "z_conv": { "weights": conv_weights(z_w),
                    "bias":    z_b.cpu().numpy().tolist() },
        "r_conv": { "weights": conv_weights(r_w),
                    "bias":    r_b.cpu().numpy().tolist() },
        "cand":   { "weights": conv_weights(gru.cand.weight.detach()),
                    "bias":    gru.cand.bias.detach().cpu().numpy().tolist() },
        "gamma":  gru.gamma.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(out),
        # Per-step reference outputs for debug (block test only checks final).
        "intermediates": {
            "cat_bh":   to_nhwc(cat_bh),
            "z_pre":    to_nhwc(z_pre),
            "z":        to_nhwc(z),
            "r_pre":    to_nhwc(r_pre),
            "r":        to_nhwc(r),
            "rh":       to_nhwc(rh),
            "cat_brh":  to_nhwc(cat_brh),
            "cand_pre": to_nhwc(cand_pre),
            "h_til":    to_nhwc(h_til),
            "h_new":    to_nhwc(h_new),
            "b_out":    to_nhwc(b_out),
        },
    }


def generate_decoder_block():
    """Realistic decoder block: deep(128,4,4) + skip(112,8,8) → (64,8,8)."""
    deep_c, skip_c, out_c = 128, 112, 64
    deep_H, deep_W = 4, 4
    skip_H, skip_W = 8, 8
    conv1 = nn.Conv2d(deep_c + skip_c, out_c, kernel_size=3, padding=1, bias=True)
    conv2 = nn.Conv2d(out_c,           out_c, kernel_size=3, padding=1, bias=True)
    deep = torch.randn(1, deep_c, deep_H, deep_W)
    skip = torch.randn(1, skip_c, skip_H, skip_W)
    with torch.no_grad():
        up   = F.interpolate(deep, scale_factor=2, mode='bilinear', align_corners=False)
        cat  = torch.cat([up, skip], dim=1)
        mid  = torch.clamp(conv1(cat), 0, 6)
        y    = torch.clamp(conv2(mid), 0, 6)
    return {
        "deep_channels": deep_c,
        "skip_channels": skip_c,
        "out_channels":  out_c,
        "deep_shape":    [1, deep_c, deep_H, deep_W],
        "skip_shape":    [1, skip_c, skip_H, skip_W],
        "output_shape":  [1, out_c,  skip_H, skip_W],
        "deep_input":    to_nhwc(deep),
        "skip_input":    to_nhwc(skip),
        "conv1": { "weights": conv_weights(conv1.weight.detach()),
                   "bias":    conv1.bias.detach().cpu().numpy().tolist() },
        "conv2": { "weights": conv_weights(conv2.weight.detach()),
                   "bias":    conv2.bias.detach().cpu().numpy().tolist() },
        "expected_output": to_nhwc(y),
    }


def dw_weights(w):
    """[C,1,KH,KW] → vec4 list for depthwise_conv2d.wgsl."""
    C, _, KH, KW = w.shape
    assert C % 4 == 0
    out = []
    for ky in range(KH):
        for kx in range(KW):
            for c in range(C // 4):
                out.extend([w[c*4+ch, 0, ky, kx].item() for ch in range(4)])
    return out


# ── Registry ──────────────────────────────────────────────────────────────────

FIXTURES = {
    "sigmoid":                  generate_sigmoid,
    "tanh":                     generate_tanh,
    "elementwise_add":          generate_elementwise_add,
    "elementwise_mul":          generate_elementwise_mul,
    "gru_update":               generate_gru_update,
    "gamma_residual":           generate_gamma_residual,
    "conv2d_1x1":               generate_conv2d_1x1,
    "conv2d_3x3":               generate_conv2d_3x3,
    "conv2d_3x3_stride2":       generate_conv2d_3x3_stride2,
    "conv2d_3x3_bn_fused":      generate_conv2d_3x3_bn_fused,
    "depthwise_3x3":            generate_depthwise_3x3,
    "bilinear_upsample_2x":     generate_bilinear_upsample,
    "channel_concat":           generate_channel_concat,
    "conv2d_add":               generate_conv2d_add,
    "upsample_concat":          generate_upsample_concat,
    "upsample_conv1x1":         generate_upsample_conv1x1,
    "upsample_sigmoid":         generate_upsample_sigmoid,
    "depthwise_separable":      generate_depthwise_separable,
    "mbconv_k3_s1_residual":    generate_mbconv_k3_s1_residual,
    "mbconv_k3_s2":             generate_mbconv_k3_s2,
    "decoder_block":            generate_decoder_block,
    "convgru_block":            generate_convgru_block,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        data = fn()
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"✓ {path}")
