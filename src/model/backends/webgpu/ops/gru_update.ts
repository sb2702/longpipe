import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import gruUpdateF32Src from "~/model/backends/webgpu/shaders/gru_update.wgsl";
import gruUpdateF16Src from "~/model/backends/webgpu/shaders/gru_update_f16.wgsl";

// Fused (1 - z) * h_prev + z * h_til. All three inputs share shape.
export class GruUpdateWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, z: Tensor, h_prev: Tensor, h_til: Tensor) {
    super(backend);
    this.shader = backend.dtype === "f16" ? gruUpdateF16Src : gruUpdateF32Src;

    const size = z.h * z.w * z.c;

    this.output = backend.tensor(z.h, z.w, z.c);
    this.inputs = [z, h_prev, h_til];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([size, 0, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(size / 256), 1, 1];
  }
}
