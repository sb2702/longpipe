import type { Tensor, MLBuffer, UpsampleConv1x1Params } from "~/model/backend";
import type { Conv2DWeights } from "~/model/weights";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import upsampleConv1x1Src from "~/model/backends/webgpu/shaders/upsample_conv1x1.wgsl";

export class UpsampleConv1x1WebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader = upsampleConv1x1Src;

  constructor(backend: WebGPUBackend, input: Tensor, w: Conv2DWeights, params: UpsampleConv1x1Params) {
    super(backend);

    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;

    this.output  = backend.tensor(params.outH, params.outW, params.outChannels);
    this.inputs  = [input];
    this.weights = [backend.upload(new Float32Array(w.weights)), backend.upload(new Float32Array(w.bias))];

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
