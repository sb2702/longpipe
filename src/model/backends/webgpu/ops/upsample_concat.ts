import type { Tensor, MLBuffer, UpsampleParams } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import upsampleConcatF32Src from "~/model/backends/webgpu/shaders/upsample_concat.wgsl";
import upsampleConcatF16Src from "~/model/backends/webgpu/shaders/upsample_concat_f16.wgsl";

export class UpsampleConcatWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, a: Tensor, b: Tensor, params: UpsampleParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? upsampleConcatF16Src : upsampleConcatF32Src;

    const aGroups   = a.c / 4;
    const bGroups   = b.c / 4;
    const outGroups = aGroups + bGroups;

    this.output = backend.tensor(params.outH, params.outW, a.c + b.c);
    this.inputs = [a, b];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      a.h, a.w, params.outH, params.outW,
      aGroups, bGroups, outGroups, 0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(params.outW / 8), Math.ceil(params.outH / 8), outGroups];
  }
}
