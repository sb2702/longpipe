import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import mulF32Src from "~/model/backends/webgpu/shaders/elementwise_mul.wgsl";
import mulF16Src from "~/model/backends/webgpu/shaders/elementwise_mul_f16.wgsl";

export class ElementwiseMulWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, a: Tensor, b: Tensor) {
    super(backend);
    this.shader = backend.dtype === "f16" ? mulF16Src : mulF32Src;

    const size = a.h * a.w * a.c;

    this.output = backend.tensor(a.h, a.w, a.c);
    this.inputs = [a, b];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([size, 0, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(size / 256), 1, 1];
  }
}
