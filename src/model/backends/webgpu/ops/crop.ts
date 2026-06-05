import type { Tensor, UpsampleParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/crop.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/crop_f16.wgsl";

export class CropWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, params: UpsampleParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    const groups = input.c / 4;
    this.output = backend.tensor(params.outH, params.outW, input.c);
    this.inputs = [input];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([input.w, params.outH, params.outW, groups]));

    this.defaultSetup();
    this.dispatch = [Math.ceil(params.outW / 8), Math.ceil(params.outH / 8), groups];
  }
}
