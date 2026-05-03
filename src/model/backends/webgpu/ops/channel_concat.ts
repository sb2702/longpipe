import type { Tensor, MLBuffer } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import channelConcatSrc from "~/model/backends/webgpu/shaders/channel_concat.wgsl";

export class ChannelConcatWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader = channelConcatSrc;

  constructor(backend: WebGPUBackend, a: Tensor, b: Tensor) {
    super(backend);

    const aGroups  = a.c / 4;
    const bGroups  = b.c / 4;
    const outGroups = aGroups + bGroups;

    this.output = backend.tensor(a.h, a.w, a.c + b.c);
    this.inputs = [a, b];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      a.h, a.w, aGroups, bGroups, outGroups, 0, 0, 0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(a.w / 8), Math.ceil(a.h / 8), outGroups];
  }
}
