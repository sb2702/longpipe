import type { Tensor } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op";
import addSrc from "~/model/backends/webgpu/shaders/add.wgsl";

export class AddWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: never[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader = addSrc;

  constructor(backend: WebGPUBackend, a: Tensor, b: Tensor) {
    super(backend);

    const size = a.h * a.w * a.c;

    this.output = backend.makeOutputTensor(a.h, a.w, a.c);
    this.inputs = [a, b];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([size, 0, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(size / 256), 1, 1];
  }
}
