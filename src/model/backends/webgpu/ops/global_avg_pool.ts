import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/global_avg_pool.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/global_avg_pool_f16.wgsl";

// (H, W, C) → (1, 1, C) mean. One workgroup per channel group.
export class GlobalAvgPoolWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;
    const g = input.c / 4;
    this.output = backend.tensor(1, 1, input.c);
    this.inputs = [input];
    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([input.h * input.w, g, 0, 0]));
    this.defaultSetup();
    this.dispatch = [g, 1, 1];
  }
}
