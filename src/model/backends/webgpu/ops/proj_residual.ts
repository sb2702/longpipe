import type { Tensor, MLBuffer, ProjResidualParams } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import projResidualF32Src from "~/model/backends/webgpu/shaders/proj_residual.wgsl";
import projResidualF16Src from "~/model/backends/webgpu/shaders/proj_residual_f16.wgsl";
import { toUploadView } from "~/utils/weights.ts";

// Bespoke 1×1 conv (no activation) + residual add. `input` is the depthwise
// output (mid channels); `skip` is the residual at out channels.
export class ProjResidualWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, skip: Tensor, w: Conv2DWeights, params: ProjResidualParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? projResidualF16Src : projResidualF32Src;

    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;

    this.output  = backend.tensor(input.h, input.w, params.outChannels);
    this.inputs  = [input, skip];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([input.h, input.w, inGroups, outGroups]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(input.w / 8), Math.ceil(input.h / 8), outGroups];
  }
}
