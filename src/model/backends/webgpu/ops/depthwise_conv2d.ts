import type { Tensor, MLBuffer, DepthwiseParams } from "~/model/backend";
import type { DepthwiseWeights } from "~/model/weights";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import depthwiseF32Src from "~/model/backends/webgpu/shaders/depthwise_conv2d.wgsl";
import depthwiseF16Src from "~/model/backends/webgpu/shaders/depthwise_conv2d_f16.wgsl";
import { convOutSize, resolvePad } from "~/model/backends/webgpu/ops/conv_utils";
import { toUploadView } from "~/utils/weights";

export class DepthwiseConv2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, w: DepthwiseWeights, params: DepthwiseParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? depthwiseF16Src : depthwiseF32Src;

    const outH = convOutSize(input.h, params.kernel, params.stride, params.padding);
    const outW = convOutSize(input.w, params.kernel, params.stride, params.padding);
    const channelGroups = input.c / 4;
    const padTop  = resolvePad(params.padding, input.h, outH, params.kernel, params.stride);
    const padLeft = resolvePad(params.padding, input.w, outW, params.kernel, params.stride);

    this.output  = backend.tensor(outH, outW, input.c);
    this.inputs  = [input];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, outH, outW,
      channelGroups,
      params.kernel, params.kernel,
      params.stride, padTop, padLeft,
      params.activation === "relu6" ? 1 : 0,
      0, // _pad0
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(outW / 8), Math.ceil(outH / 8), channelGroups];
  }
}
