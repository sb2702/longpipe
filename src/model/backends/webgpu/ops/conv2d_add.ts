import type { Tensor, MLBuffer, Conv2dParams } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import conv2dAddF32Src from "~/model/backends/webgpu/shaders/conv2d_add.wgsl";
import conv2dAddF16Src from "~/model/backends/webgpu/shaders/conv2d_add_f16.wgsl";
import { convOutSize, resolvePad } from "~/model/backends/webgpu/ops/conv_utils.ts";
import { toUploadView } from "~/utils/weights.ts";

export class Conv2dAddWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, skip: Tensor, w: Conv2DWeights, params: Conv2dParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? conv2dAddF16Src : conv2dAddF32Src;

    const outH = convOutSize(input.h, params.kernel, params.stride, params.padding);
    const outW = convOutSize(input.w, params.kernel, params.stride, params.padding);
    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;
    const padTop  = resolvePad(params.padding, input.h, outH, params.kernel, params.stride);
    const padLeft = resolvePad(params.padding, input.w, outW, params.kernel, params.stride);

    this.output  = backend.tensor(outH, outW, params.outChannels);
    this.inputs  = [input, skip];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, outH, outW,
      inGroups, outGroups,
      params.kernel, params.kernel,
      params.stride, padTop, padLeft,
      params.activation === "relu6" ? 1 : 0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(outW / 8), Math.ceil(outH / 8), outGroups];
  }
}
