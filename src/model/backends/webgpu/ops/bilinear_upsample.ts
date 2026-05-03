import type { Tensor, MLBuffer, UpsampleParams } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import bilinearSrc from "~/model/backends/webgpu/shaders/bilinear_upsample.wgsl";

export class BilinearUpsampleWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader = bilinearSrc;

  constructor(backend: WebGPUBackend, input: Tensor, params: UpsampleParams) {
    super(backend);

    const channelGroups = input.c / 4;

    this.output = backend.tensor(params.outH, params.outW, input.c);
    this.inputs = [input];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, params.outH, params.outW,
      channelGroups, 0, 0, 0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(params.outW / 8), Math.ceil(params.outH / 8), channelGroups];
  }
}
