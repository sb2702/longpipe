import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import { toUploadView } from "~/utils/weights.ts";
import gammaF32Src from "~/model/backends/webgpu/shaders/gamma_residual.wgsl";
import gammaF16Src from "~/model/backends/webgpu/shaders/gamma_residual_f16.wgsl";

// b_out = b + γ ⊙ h_new. γ is one f32 per channel (length = b.c);
// uploaded once at construction as a weight.
export class GammaResidualWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, b: Tensor, h_new: Tensor, gamma: ArrayLike<number>) {
    super(backend);
    this.shader = backend.dtype === "f16" ? gammaF16Src : gammaF32Src;

    const channelGroups = b.c / 4;

    this.output  = backend.tensor(b.h, b.w, b.c);
    this.inputs  = [b, h_new];
    this.weights = [backend.upload(toUploadView(gamma))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([b.h, b.w, channelGroups, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(b.w / 8), Math.ceil(b.h / 8), channelGroups];
  }
}
