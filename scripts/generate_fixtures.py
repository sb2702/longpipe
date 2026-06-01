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


def generate_proj_residual():
    # Bespoke 1×1 proj + residual add. `input` is the depthwise output (mid
    # channels), `skip` is the residual at out channels. Distinct in/out groups
    # (16→8) to exercise the o*I+i weight indexing.
    Cmid, Cout, H, W = 16, 8, 8, 8
    conv = nn.Conv2d(Cmid, Cout, kernel_size=1, stride=1, padding=0, bias=True)
    x    = torch.randn(1, Cmid, H, W)
    skip = torch.randn(1, Cout, H, W)
    with torch.no_grad():
        y = conv(x) + skip
    return {
        "in_channels":  Cmid,
        "out_channels": Cout,
        "input_shape":  [1, Cmid, H, W],
        "input":   to_nhwc(x),
        "skip":    to_nhwc(skip),
        "weights": conv_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(y),
    }


def conv_expand_weights(w):
    """[2, I, KH, KW] (I mult of 4) → 9 * in_groups mat4x2, 8 floats each
    (col-major c0r0,c0r1,...,c3r0,c3r1) for conv_expand."""
    O, I, KH, KW = w.shape
    assert O == 2 and I % 4 == 0
    in_groups = I // 4
    out = []
    for ky in range(KH):
        for kx in range(KW):
            for ig in range(in_groups):
                for col in range(4):
                    in_idx = ig * 4 + col
                    for row in range(2):
                        out.append(float(w[row, in_idx, ky, kx]))
    return out


def generate_conv_expand():
    # Bespoke N→2 conv 3×3 (pad 1) + relu (wrapper expand_feat). Input mult of 4.
    # Output packed to 4 channels (.xy = the 2 native outputs, .zw = 0).
    in_c, H, W = 16, 8, 8
    conv = nn.Conv2d(in_c, 2, kernel_size=3, padding=1, bias=True)
    x = torch.randn(1, in_c, H, W)
    with torch.no_grad():
        y2 = F.relu(conv(x))                       # (1, 2, H, W)
    y4 = torch.cat([y2, torch.zeros(1, 2, H, W)], dim=1)
    return {
        "in_channels": in_c,
        "input_shape": [1, in_c, H, W],
        "input":   to_nhwc(x),
        "weights": conv_expand_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),   # 2 floats
        "expected_output": to_nhwc(y4),
    }


def cat_conv_6to2_weights(w):
    """[2, 6, KH, KW] → 9 * 2 mat3x2, 6 floats each (col-major
    c0r0,c0r1,c1r0,c1r1,c2r0,c2r1). Input groups (0,1,2) then (3,4,5)."""
    O, I, KH, KW = w.shape
    assert O == 2 and I == 6
    out = []
    for ky in range(KH):
        for kx in range(KW):
            for ig in range(2):
                for col in range(3):
                    in_idx = ig * 3 + col
                    for row in range(2):
                        out.append(float(w[row, in_idx, ky, kx]))
    return out


def generate_cat_conv_6to2():
    # Fused concat(u[2], d[4]) → 6→2 conv 3×3 (pad 1) + relu (E up1_combine).
    # u packed in 4ch (.xy), d full 4ch. Output 4ch carrier (.xy). Conv in-channel
    # order = [u.0, u.1, d.0, d.1, d.2, d.3].
    H, W = 8, 8
    conv = nn.Conv2d(6, 2, kernel_size=3, padding=1, bias=True)
    u2 = torch.randn(1, 2, H, W)
    d4 = torch.randn(1, 4, H, W)
    with torch.no_grad():
        y2 = F.relu(conv(torch.cat([u2, d4], dim=1)))
    u4 = torch.cat([u2, torch.zeros(1, 2, H, W)], dim=1)
    y4 = torch.cat([y2, torch.zeros(1, 2, H, W)], dim=1)
    return {
        "in_h": H, "in_w": W,
        "u_in":    to_nhwc(u4),
        "d_in":    to_nhwc(d4),
        "weights": cat_conv_6to2_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),   # 2 floats
        "expected_output": to_nhwc(y4),
    }


def up_final_weights(w):
    """[1,5,3,3] → 18 vec4: [0..8]=(w0,w1,0,0) u; [9..17]=(w2,w3,w4,0) rgb."""
    O, I, KH, KW = w.shape
    assert O == 1 and I == 5
    out = []
    for ky in range(3):
        for kx in range(3):
            out.extend([float(w[0, 0, ky, kx]), float(w[0, 1, ky, kx]), 0.0, 0.0])
    for ky in range(3):
        for kx in range(3):
            out.extend([float(w[0, 2, ky, kx]), float(w[0, 3, ky, kx]),
                        float(w[0, 4, ky, kx]), 0.0])
    return out


def up_final_skip_weights(w):
    """[1,9,3,3] → 27 vec4 (3 per kpos): (w0,w1,0,0)|(w2..w5)|(w6,w7,w8,0)."""
    O, I, KH, KW = w.shape
    assert O == 1 and I == 9
    out = []
    for ky in range(3):
        for kx in range(3):
            out.extend([float(w[0, 0, ky, kx]), float(w[0, 1, ky, kx]), 0.0, 0.0])
            out.extend([float(w[0, 2, ky, kx]), float(w[0, 3, ky, kx]),
                        float(w[0, 4, ky, kx]), float(w[0, 5, ky, kx])])
            out.extend([float(w[0, 6, ky, kx]), float(w[0, 7, ky, kx]),
                        float(w[0, 8, ky, kx]), 0.0])
    return out


def generate_up_final():
    # A/B alpha head: concat(u[2], rgb[3]) → conv 5→1 → sigmoid. Output .x = alpha.
    H, W = 8, 8
    conv = nn.Conv2d(5, 1, kernel_size=3, padding=1, bias=True)
    u2  = torch.randn(1, 2, H, W)
    rgb = torch.randn(1, 3, H, W)
    with torch.no_grad():
        y = torch.sigmoid(conv(torch.cat([u2, rgb], dim=1)))
    return {
        "in_h": H, "in_w": W,
        "u_in":    to_nhwc(torch.cat([u2,  torch.zeros(1, 2, H, W)], dim=1)),
        "rgb_in":  to_nhwc(torch.cat([rgb, torch.zeros(1, 1, H, W)], dim=1)),
        "weights": up_final_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),   # 1
        "expected_output": to_nhwc(torch.cat([y, torch.zeros(1, 3, H, W)], dim=1)),
    }


def generate_up_final_skip():
    # C/D alpha head: concat(u[2], d_full[4], rgb[3]) → conv 9→1 → sigmoid.
    H, W = 8, 8
    conv = nn.Conv2d(9, 1, kernel_size=3, padding=1, bias=True)
    u2  = torch.randn(1, 2, H, W)
    d4  = torch.randn(1, 4, H, W)
    rgb = torch.randn(1, 3, H, W)
    with torch.no_grad():
        y = torch.sigmoid(conv(torch.cat([u2, d4, rgb], dim=1)))
    return {
        "in_h": H, "in_w": W,
        "u_in":    to_nhwc(torch.cat([u2,  torch.zeros(1, 2, H, W)], dim=1)),
        "d_in":    to_nhwc(d4),
        "rgb_in":  to_nhwc(torch.cat([rgb, torch.zeros(1, 1, H, W)], dim=1)),
        "weights": up_final_skip_weights(conv.weight.detach()),
        "bias":    conv.bias.detach().cpu().numpy().tolist(),
        "expected_output": to_nhwc(torch.cat([y, torch.zeros(1, 3, H, W)], dim=1)),
    }


def generate_down_adapter():
    # Fused stride-N 3×3 conv (4→4, relu) + 1×1 adapter (4→3) (E down2+adapter).
    # Symmetric pad 1. Output 4ch (.xyz = adapter, .w = 0). Adapter weight padded
    # [3,4]→[4,4] (4th out row 0); adapter bias [3]→[4].
    in_c, H, W, stride = 4, 16, 16, 2
    down  = nn.Conv2d(in_c, 4, kernel_size=3, stride=stride, padding=1, bias=True)
    adapt = nn.Conv2d(4, 3, kernel_size=1, bias=True)
    x = torch.randn(1, in_c, H, W)
    with torch.no_grad():
        d = F.relu(down(x))
        a = adapt(d)
    oh, ow = a.shape[2], a.shape[3]
    a4 = torch.cat([a, torch.zeros(1, 1, oh, ow)], dim=1)   # .xyz = adapter, .w = 0

    adapt_w_pad = torch.zeros(4, 4, 1, 1)
    adapt_w_pad[:3] = adapt.weight.detach()
    adapt_b_pad = torch.zeros(4)
    adapt_b_pad[:3] = adapt.bias.detach()

    return {
        "in_channels": in_c, "stride": stride,
        "input_shape":  [1, in_c, H, W],
        "output_shape": [1, 4, oh, ow],
        "input":   to_nhwc(x),
        "down_weights":  conv_weights(down.weight.detach()),   # 9 mat4x4
        "down_bias":     down.bias.detach().cpu().numpy().tolist(),   # 4
        "adapt_weights": conv_weights(adapt_w_pad),            # 1 mat4x4
        "adapt_bias":    adapt_b_pad.cpu().numpy().tolist(),   # 4 (.xyz used)
        "expected_output": to_nhwc(a4),
    }


def generate_gru_fused():
    """Production ConvGRU (c_up=2, split_ratio=0.5 → passthrough=1, recurrent=1),
    fused gates + cand_update. Inputs/outputs packed to 4 channels (only the
    leading lanes carry data). Weights use the fused shaders' special packing
    (9 vec4 per kpos), NOT the mat4x4 conv layout."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from models.temporal_model import ConvGRU

    H, W = 8, 8
    gru = ConvGRU(channels=2, kernel=3, split_ratio=0.5).eval()
    assert gru.recurrent_ch == 1 and gru.passthrough_ch == 1
    with torch.no_grad():
        nn.init.uniform_(gru.gates.weight, -0.5, 0.5)
        nn.init.uniform_(gru.gates.bias,   -0.2, 0.2)
        nn.init.kaiming_normal_(gru.cand.weight)
        nn.init.uniform_(gru.cand.bias,    -0.2, 0.2)
        gru.gamma.copy_(torch.randn_like(gru.gamma))

    x      = torch.randn(1, 2, H, W)
    h_prev = torch.randn(1, 1, H, W)
    a = x[:, :1]
    b = x[:, 1:]

    with torch.no_grad():
        zr    = gru.gates(torch.cat([b, h_prev], dim=1))
        z     = torch.sigmoid(zr[:, :1])
        r     = torch.sigmoid(zr[:, 1:])
        rh    = r * h_prev
        h_til = torch.tanh(gru.cand(torch.cat([b, rh], dim=1)))
        h_new = (1.0 - z) * h_prev + z * h_til
        b_out = b + gru.gamma.view(1, -1, 1, 1) * h_new

        # Hidden state lives in channel .z of the carrier tensor: the GRU output
        # (a, b_out, h_new, 0) is fed back unchanged as next frame's h_prev, so
        # both read h_prev from .z and write h_new to .z (option A — no extra buf).
        zeros       = torch.zeros_like(a)
        u_in_4      = torch.cat([a, b, zeros, zeros], dim=1)
        h_prev_4    = torch.cat([zeros, zeros, h_prev, zeros], dim=1)
        gates_out_4 = torch.cat([z, r, zeros, zeros], dim=1)
        out_4       = torch.cat([a, b_out, h_new, zeros], dim=1)

    # gates: 9 vec4 = (z_w_b, z_w_h, r_w_b, r_w_h); out0=z,out1=r; in0=b,in1=h.
    gw = gru.gates.weight.detach()
    gates_packed = []
    for ky in range(3):
        for kx in range(3):
            gates_packed += [float(gw[0, 0, ky, kx]), float(gw[0, 1, ky, kx]),
                             float(gw[1, 0, ky, kx]), float(gw[1, 1, ky, kx])]
    # cand: 9 vec4 (.xy = b_w, rh_w); in0=b, in1=rh.
    cw = gru.cand.weight.detach()
    cand_packed = []
    for ky in range(3):
        for kx in range(3):
            cand_packed += [float(cw[0, 0, ky, kx]), float(cw[0, 1, ky, kx]), 0.0, 0.0]

    return {
        "height": H, "width": W,
        "u_in":          to_nhwc(u_in_4),
        "h_prev":        to_nhwc(h_prev_4),
        "gates_weights": gates_packed,
        "gates_bias":    gru.gates.bias.detach().cpu().numpy().tolist(),  # [z_bias, r_bias]
        "cand_weights":  cand_packed,
        "cand_bias":     gru.cand.bias.detach().cpu().numpy().tolist(),   # [cand_bias]
        "gamma":         gru.gamma.detach().cpu().numpy().tolist(),       # [gamma]
        "expected_gates":  to_nhwc(gates_out_4),
        "expected_output": to_nhwc(out_4),
    }


def generate_concat_conv2d():
    # Fused concat(a, b) → 3×3 conv (pad 1) → relu6. Both inputs at output
    # resolution; conv in-channels ordered [a, b]. Distinct a/b groups (16, 8).
    a_c, b_c, out_c, H, W = 16, 8, 8, 8, 8
    conv = nn.Conv2d(a_c + b_c, out_c, kernel_size=3, padding=1, bias=True)
    a = torch.randn(1, a_c, H, W)
    b = torch.randn(1, b_c, H, W)
    with torch.no_grad():
        y = F.relu6(conv(torch.cat([a, b], dim=1)))
    return {
        "a_channels":   a_c,
        "b_channels":   b_c,
        "out_channels": out_c,
        "in_h": H, "in_w": W,
        "a_shape": [1, a_c, H, W],
        "b_shape": [1, b_c, H, W],
        "input_a": to_nhwc(a),
        "input_b": to_nhwc(b),
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


def _pad4(n: int) -> int:
    return ((n + 3) // 4) * 4


def _to_nhwc_padded(t: torch.Tensor):
    """NCHW → NHWC vec4 with channels padded to multiple of 4 (zeros)."""
    nhwc = t.permute(0, 2, 3, 1).contiguous()
    B, H, W, C = nhwc.shape
    if C % 4 != 0:
        nhwc = F.pad(nhwc, (0, 4 - (C % 4)), value=0.0)
        C = nhwc.shape[-1]
    return nhwc.reshape(B, H, W, C // 4, 4).flatten().cpu().numpy().tolist()


def _pad_conv(w: torch.Tensor, b: torch.Tensor):
    """Pad conv [O,I,KH,KW] + bias [O] so both I and O are multiples of 4.
    Numerically identical to the unpadded conv as long as upstream-padded
    inputs zero-fill and downstream consumers ignore the extra outputs."""
    O, I, KH, KW = w.shape
    tI, tO = _pad4(I), _pad4(O)
    if I != tI:
        w = torch.cat([w, torch.zeros(O, tI - I, KH, KW)], dim=1)
    if O != tO:
        w = torch.cat([w, torch.zeros(tO - O, tI, KH, KW)], dim=0)
        b = torch.cat([b, torch.zeros(tO - O)], dim=0)
    return w, b


def _pack_conv(conv: nn.Conv2d):
    w, b = _pad_conv(conv.weight.detach().clone(), conv.bias.detach().clone())
    return { "weights": conv_weights(w), "bias": b.cpu().numpy().tolist() }


def _pack_conv_expand(conv: nn.Conv2d):
    return { "weights": conv_expand_weights(conv.weight.detach()),
             "bias": conv.bias.detach().cpu().numpy().tolist() }


def _pack_cat_conv_6to2(conv: nn.Conv2d):
    return { "weights": cat_conv_6to2_weights(conv.weight.detach()),
             "bias": conv.bias.detach().cpu().numpy().tolist() }


def _pack_up_final(conv: nn.Conv2d):
    return { "weights": up_final_weights(conv.weight.detach()),
             "bias": conv.bias.detach().cpu().numpy().tolist() }


def _pack_up_final_skip(conv: nn.Conv2d):
    return { "weights": up_final_skip_weights(conv.weight.detach()),
             "bias": conv.bias.detach().cpu().numpy().tolist() }


class _WrapperStubBase(nn.Module):
    """UNetMattingModel only reads output_conv.in_channels at __init__; the
    fixture feeds feat_lr directly so forward_features is never invoked."""
    def __init__(self, feat_ch: int):
        super().__init__()
        self.output_conv = nn.Conv2d(feat_ch, 1, 1)


def _build_wrapper_fixture(variant: str, base_hw: int, c_high: int,
                           c_low: int, feat_ch: int):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from models.unet_model import UNetMattingModel, VARIANT_SPEC

    canvas = int(round(base_hw * VARIANT_SPEC[variant]['canvas_mul']))

    base = _WrapperStubBase(feat_ch)
    wrapper = UNetMattingModel(base, variant=variant, c_high=c_high, c_low=c_low).eval()
    # Force gru_output to identity-at-init so the static-only block runner
    # can omit it without affecting numerics.
    nn.init.zeros_(wrapper.gru_output.cand.weight)
    nn.init.zeros_(wrapper.gru_output.cand.bias)

    x_hr    = torch.randn(1, 3, canvas, canvas)
    feat_lr = torch.randn(1, feat_ch, base_hw, base_hw)
    with torch.no_grad():
        adapted, inter = wrapper._down_path(x_hr)
        alpha, _ = wrapper._up_path(x_hr, feat_lr, inter, h_gru=None)

    out = {
        "variant":   variant,
        "c_high":    c_high,
        "c_low":     c_low,
        "c_up":      wrapper.c_up,
        "feat_ch":   feat_ch,
        "base_hw":   base_hw,
        "canvas":    canvas,
        "x_hr":      _to_nhwc_padded(x_hr),        # 3 channels padded to 4
        "feat_lr":   to_nhwc(feat_lr),             # already mult-of-4
        "down1":     _pack_conv(wrapper.down1),    # mat4x4 (B: DownAdapter; E/D: Conv2d)
        "adapter":   _pack_conv(wrapper.adapter),  # mat4x4 padded 4→3
        "expandFeat":_pack_conv_expand(wrapper.expand_feat),   # mat4x2 N→2
        # Single-channel flat alpha (test extracts channel 0 from sigmoid output).
        "expected_output": alpha.squeeze(0).squeeze(0).flatten().cpu().numpy().tolist(),
    }
    if wrapper.one_stage:
        out["upCombine"] = _pack_up_final(wrapper.up_combine)            # 5→1
    else:
        out["down2"]      = _pack_conv(wrapper.down2)                    # mat4x4
        out["up1Combine"] = _pack_cat_conv_6to2(wrapper.up1_combine)     # mat3x2 6→2
        out["upCombine"]  = (_pack_up_final_skip(wrapper.up2_combine)    # D: 9→1
                             if wrapper.has_input_skip
                             else _pack_up_final(wrapper.up2_combine))   # E: 5→1
    return out


def generate_wrapper_a():
    return _build_wrapper_fixture('A', base_hw=8, c_high=4, c_low=4, feat_ch=8)


def generate_wrapper_b():
    # Production widths: c_high=c_low=4 → c_up=2 (the native narrow path).
    return _build_wrapper_fixture('B', base_hw=8, c_high=4, c_low=4, feat_ch=8)


def generate_wrapper_e():
    return _build_wrapper_fixture('E', base_hw=8, c_high=4, c_low=4, feat_ch=8)


def generate_wrapper_d():
    return _build_wrapper_fixture('D', base_hw=8, c_high=4, c_low=4, feat_ch=8)


# ── Multi-frame tier fixtures (5 production tiers, T frames each) ─────────

class _TierStubBase(nn.Module):
    """Stand-in for MattingModel with a deterministic feat_lr producer.
    Per-frame feat_lr varies with x_hr → adapter chain, so multi-frame
    inputs propagate as they would with the real base."""
    def __init__(self, feat_ch: int):
        super().__init__()
        self.output_conv = nn.Conv2d(feat_ch, 1, 1)
        self.feat_conv   = nn.Conv2d(3, feat_ch, 3, padding=1)

    def forward_features(self, x, hiddens=None):
        return self.feat_conv(x), hiddens


def _build_tier_fixture(variant: str, base_hw: int, c_high: int, c_low: int,
                        feat_ch: int, T: int = 3):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from models.unet_model import UNetMattingModel, VARIANT_SPEC
    from models.temporal_model import ConvGRU as _ConvGRU

    canvas = int(round(base_hw * VARIANT_SPEC[variant]['canvas_mul']))

    base = _TierStubBase(feat_ch)
    wrapper = UNetMattingModel(base, variant=variant, c_high=c_high, c_low=c_low).eval()
    # Replace gru_output with split_ratio=1.0 so passthrough=0 and all c_up
    # channels are recurrent — avoids the recurrent_ch=c_up/2 mult-of-4 issue
    # at c_up=4. Cand is non-zero so state-threading actually matters.
    wrapper.gru_output = _ConvGRU(wrapper.c_up, split_ratio=1.0).eval()
    with torch.no_grad():
        wrapper.gru_output.cand.weight.normal_(0, 0.02)
        wrapper.gru_output.cand.bias.zero_()
        wrapper.gru_output.gamma.copy_(torch.randn_like(wrapper.gru_output.gamma) * 0.5)

    xs       = [torch.randn(1, 3, canvas, canvas) for _ in range(T)]
    feat_lrs = []
    alphas   = []
    h_gru, base_hidden = None, None
    with torch.no_grad():
        for x in xs:
            adapted, inter      = wrapper._down_path(x)
            feat_lr, base_hidden = wrapper.base.forward_features(adapted, base_hidden)
            alpha, h_gru        = wrapper._up_path(x, feat_lr, inter, h_gru=h_gru)
            feat_lrs.append(feat_lr)
            alphas.append(alpha)

    # gru_output gates split into z_conv + r_conv (same trick as ConvGRU block).
    gru = wrapper.gru_output
    gates_w = gru.gates.weight.detach()
    gates_b = gru.gates.bias.detach()
    z_w, r_w = gates_w[:gru.recurrent_ch], gates_w[gru.recurrent_ch:]
    z_b, r_b = gates_b[:gru.recurrent_ch], gates_b[gru.recurrent_ch:]

    out = {
        "variant":   variant,
        "c_high":    c_high,
        "c_low":     c_low,
        "c_up":      wrapper.c_up,
        "feat_ch":   feat_ch,
        "base_hw":   base_hw,
        "canvas":    canvas,
        "frames":    T,
        "gru_passthrough": gru.passthrough_ch,   # expected 0 with split_ratio=1.0
        "gru_recurrent":   gru.recurrent_ch,
        # Per-frame inputs.
        "x_hr_per_frame":    [_to_nhwc_padded(x) for x in xs],
        "feat_lr_per_frame": [to_nhwc(f) for f in feat_lrs],
        # Wrapper weights (frame-invariant).
        "down1":      _pack_conv(wrapper.down1),
        "adapter":    _pack_conv(wrapper.adapter),
        "expandFeat": _pack_conv(wrapper.expand_feat),
        # gru_output weights (gates pre-split into z_conv + r_conv).
        "gruOutput": {
            "zConv": { "weights": conv_weights(z_w),
                       "bias":    z_b.cpu().numpy().tolist() },
            "rConv": { "weights": conv_weights(r_w),
                       "bias":    r_b.cpu().numpy().tolist() },
            "cand":  { "weights": conv_weights(gru.cand.weight.detach()),
                       "bias":    gru.cand.bias.detach().cpu().numpy().tolist() },
            "gamma": gru.gamma.detach().cpu().numpy().tolist(),
        },
        # Per-frame reference alphas (single-channel flat, H*W).
        "expected_alphas": [a.squeeze(0).squeeze(0).flatten().cpu().numpy().tolist()
                            for a in alphas],
    }
    if wrapper.one_stage:
        out["upCombine"] = _pack_conv(wrapper.up_combine)
    else:
        out["down2"]      = _pack_conv(wrapper.down2)
        out["up1Combine"] = _pack_conv(wrapper.up1_combine)
        out["upCombine"]  = _pack_conv(wrapper.up2_combine)
    return out


# Production tier matrix: feat_ch=32 for xl (xl decoder), 16 for the rest
# (standard decoder). Wrapper variant per the production tier spec.
def generate_tier_xl():     return _build_tier_fixture('E', 8, 8, 8, feat_ch=32)
def generate_tier_large():  return _build_tier_fixture('E', 8, 8, 8, feat_ch=16)
def generate_tier_medium(): return _build_tier_fixture('B', 8, 8, 8, feat_ch=16)
def generate_tier_small():  return _build_tier_fixture('B', 8, 8, 8, feat_ch=16)
def generate_tier_xs():     return _build_tier_fixture('E', 8, 8, 8, feat_ch=16)


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
    "conv2d_1x1":               generate_conv2d_1x1,
    "conv2d_3x3":               generate_conv2d_3x3,
    "conv2d_3x3_stride2":       generate_conv2d_3x3_stride2,
    "conv2d_3x3_bn_fused":      generate_conv2d_3x3_bn_fused,
    "depthwise_3x3":            generate_depthwise_3x3,
    "bilinear_upsample_2x":     generate_bilinear_upsample,
    "channel_concat":           generate_channel_concat,
    "conv2d_add":               generate_conv2d_add,
    "proj_residual":            generate_proj_residual,
    "concat_conv2d":            generate_concat_conv2d,
    "gru_fused":                generate_gru_fused,
    "conv_expand":              generate_conv_expand,
    "cat_conv_6to2":            generate_cat_conv_6to2,
    "down_adapter":             generate_down_adapter,
    "up_final":                 generate_up_final,
    "up_final_skip":            generate_up_final_skip,
    "upsample_concat":          generate_upsample_concat,
    "upsample_conv1x1":         generate_upsample_conv1x1,
    "upsample_sigmoid":         generate_upsample_sigmoid,
    "depthwise_separable":      generate_depthwise_separable,
    "mbconv_k3_s1_residual":    generate_mbconv_k3_s1_residual,
    "mbconv_k3_s2":             generate_mbconv_k3_s2,
    "decoder_block":            generate_decoder_block,
    "wrapper_a":                generate_wrapper_a,
    "wrapper_b":                generate_wrapper_b,
    "wrapper_e":                generate_wrapper_e,
    "wrapper_d":                generate_wrapper_d,
    "tier_xl":                  generate_tier_xl,
    "tier_large":               generate_tier_large,
    "tier_medium":              generate_tier_medium,
    "tier_small":               generate_tier_small,
    "tier_xs":                  generate_tier_xs,
}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in FIXTURES.items():
        data = fn()
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"✓ {path}")
