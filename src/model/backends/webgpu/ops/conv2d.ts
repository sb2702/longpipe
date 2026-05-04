import type { Tensor, Conv2dParams } from "~/model/backend";
import type { Conv2DWeights } from "~/model/weights";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import conv2dF32Src from "~/model/backends/webgpu/shaders/conv2d.wgsl";
import conv2dF16Src from "~/model/backends/webgpu/shaders/conv2d_f16.wgsl";
import { convOutSize, resolvePad } from "~/model/backends/webgpu/ops/conv_utils";
import { toUploadView } from "~/utils/weights";

export class Conv2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: import("~/model/backend").MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, w: Conv2DWeights, params: Conv2dParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? conv2dF16Src : conv2dF32Src;

    const outH = convOutSize(input.h, params.kernel, params.stride, params.padding);
    const outW = convOutSize(input.w, params.kernel, params.stride, params.padding);
    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;
    const padTop  = resolvePad(params.padding, input.h, outH, params.kernel, params.stride);
    const padLeft = resolvePad(params.padding, input.w, outW, params.kernel, params.stride);

    this.output  = backend.tensor(outH, outW, params.outChannels);
    this.inputs  = [input];
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
