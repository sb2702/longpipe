import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/channel_scale.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/channel_scale_f16.wgsl";

// x (H, W, C) * scale (1, 1, C), broadcast over H×W.
export class ChannelScaleWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, x: Tensor, scale: Tensor) {
    super(backend);
    if (scale.h !== 1 || scale.w !== 1 || scale.c !== x.c)
      throw new Error(`ChannelScale: scale must be 1×1×${x.c}, got ${scale.h}×${scale.w}×${scale.c}`);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;
    const g = x.c / 4;
    const n = x.h * x.w * g;
    this.output = backend.tensor(x.h, x.w, x.c);
    this.inputs = [x, scale];
    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([n, g, 0, 0]));
    this.defaultSetup();
    this.dispatch = [Math.ceil(n / 256), 1, 1];
  }
}
