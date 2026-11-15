import type { Activation } from "~/model/backend.ts";

export function convOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: number | "same" | "valid",
): number {
  if (typeof padding === "number") return Math.floor((inSize + 2 * padding - kernel) / stride) + 1;
  if (padding === "same") return Math.ceil(inSize / stride);
  return Math.floor((inSize - kernel) / stride) + 1;
}

// ConvTranspose2d output size (output_padding = 0). Inverse of a strided conv:
// k4/s2/p1 doubles the spatial dims exactly.
export function convTransposeOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: number,
): number {
  return (inSize - 1) * stride - 2 * padding + kernel;
}

export function samePadHalf(
  inSize: number,
  outSize: number,
  kernel: number,
  stride: number,
): number {
  return Math.floor(Math.max((outSize - 1) * stride + kernel - inSize, 0) / 2);
}

export function resolvePad(
  padding: number | "same" | "valid",
  inSize: number,
  outSize: number,
  kernel: number,
  stride: number,
): number {
  if (typeof padding === "number") return padding;
  if (padding === "same") return samePadHalf(inSize, outSize, kernel, stride);
  return 0;
}

// Shader activation code shared by the Conv2d / DepthwiseConv2d shaders (both
// backends): 0 none, 1 relu6, 2 relu, 3 leaky(0.1), 4 hswish, 5 sigmoid.
export function activationCode(a: Activation): number {
  switch (a) {
    case "relu6":   return 1;
    case "relu":    return 2;
    case "leaky":   return 3;
    case "hswish":  return 4;
    case "sigmoid": return 5;
    default:        return 0;
  }
}
