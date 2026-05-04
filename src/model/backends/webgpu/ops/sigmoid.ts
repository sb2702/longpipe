import type { Tensor, MLBuffer } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import sigmoidF32Src from "~/model/backends/webgpu/shaders/sigmoid.wgsl";
import sigmoidF16Src from "~/model/backends/webgpu/shaders/sigmoid_f16.wgsl";

export class SigmoidWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor) {
    super(backend);
    this.shader = backend.dtype === "f16" ? sigmoidF16Src : sigmoidF32Src;

    const nGroups = input.h * input.w * (input.c / 4);

    this.output = backend.tensor(input.h, input.w, input.c);
    this.inputs = [input];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([nGroups, 0, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(nGroups / 256), 1, 1];
  }
}
