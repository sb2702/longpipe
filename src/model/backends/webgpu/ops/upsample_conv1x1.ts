import type { Tensor, MLBuffer, UpsampleConv1x1Params } from "~/model/backend";
import type { Conv2DWeights } from "~/model/weights";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import upsampleConv1x1F32Src from "~/model/backends/webgpu/shaders/upsample_conv1x1.wgsl";
import upsampleConv1x1F16Src from "~/model/backends/webgpu/shaders/upsample_conv1x1_f16.wgsl";
import { toUploadView } from "~/utils/weights";

export class UpsampleConv1x1WebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, w: Conv2DWeights, params: UpsampleConv1x1Params) {
    super(backend);
    this.shader = backend.dtype === "f16" ? upsampleConv1x1F16Src : upsampleConv1x1F32Src;

    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;

    this.output  = backend.tensor(params.outH, params.outW, params.outChannels);
    this.inputs  = [input];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, params.outH, params.outW,
      inGroups, outGroups,
      params.activation === "relu6" ? 1 : 0,
      0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(params.outW / 8), Math.ceil(params.outH / 8), outGroups];
  }
}
