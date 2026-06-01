import type { Tensor, MLBuffer, ConcatConv2dParams } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import concatConv2dF32Src from "~/model/backends/webgpu/shaders/concat_conv2d.wgsl";
import concatConv2dF16Src from "~/model/backends/webgpu/shaders/concat_conv2d_f16.wgsl";
import { toUploadView } from "~/utils/weights.ts";

// Fused concat(a, b) → 3×3 conv (pad 1) → relu6. `a` and `b` must share the
// same spatial resolution; the conv weight's input channels are ordered [a, b].
export class ConcatConv2dWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, a: Tensor, b: Tensor, w: Conv2DWeights, params: ConcatConv2dParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? concatConv2dF16Src : concatConv2dF32Src;

    const aGroups   = a.c / 4;
    const bGroups   = b.c / 4;
    const outGroups = params.outChannels / 4;

    this.output  = backend.tensor(a.h, a.w, params.outChannels);
    this.inputs  = [a, b];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([a.h, a.w, aGroups, bGroups, outGroups, 0, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(a.w / 8), Math.ceil(a.h / 8), outGroups];
  }
}
