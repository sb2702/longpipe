import type { Tensor, DepthwiseParams } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import depthwiseSrc from "~/model/backends/webgpu/shaders/depthwise_conv2d.wgsl";
import { convOutSize, resolvePad } from "~/model/backends/webgpu/ops/conv_utils";

export class DepthwiseConv2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader = depthwiseSrc;

  constructor(
    backend: WebGPUBackend,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    params: DepthwiseParams,
  ) {
    super(backend);

    const outH = convOutSize(input.h, params.kernel, params.stride, params.padding);
    const outW = convOutSize(input.w, params.kernel, params.stride, params.padding);
    const channelGroups = input.c / 4;
    const padTop  = resolvePad(params.padding, input.h, outH, params.kernel, params.stride);
    const padLeft = resolvePad(params.padding, input.w, outW, params.kernel, params.stride);

    this.output = backend.makeOutputTensor(outH, outW, input.c);
    this.inputs = [input, weights, bias];

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
