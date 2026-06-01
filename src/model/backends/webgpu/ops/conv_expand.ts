import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import convExpandF32Src from "~/model/backends/webgpu/shaders/conv_expand.wgsl";
import convExpandF16Src from "~/model/backends/webgpu/shaders/conv_expand_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// Bespoke N→2 conv 3×3 (pad 1) + relu (wrapper expand_feat). Output is a
// 4-channel carrier tensor with the 2 native channels in .xy (.zw = 0).
// weights.weights = 9 * in_groups mat4x2 (8 floats each); weights.bias = 2.
export class ConvExpandWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, w: Conv2DWeights) {
    super(backend);
    this.shader = backend.dtype === "f16" ? convExpandF16Src : convExpandF32Src;

    const inGroups = input.c / 4;

    this.output  = backend.tensor(input.h, input.w, 4);
    this.inputs  = [input];
    this.weights = [
      backend.upload(toUploadView(w.weights)),
      backend.upload(padToVec4(w.bias)),
    ];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([input.h, input.w, inGroups, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(input.w / 8), Math.ceil(input.h / 8), 1];
  }
}
