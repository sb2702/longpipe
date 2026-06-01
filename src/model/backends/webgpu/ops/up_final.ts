import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import upFinalF32Src from "~/model/backends/webgpu/shaders/up_final.wgsl";
import upFinalF16Src from "~/model/backends/webgpu/shaders/up_final_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// A/B alpha head: fused concat(u, rgb) → conv 3×3 5→1 → sigmoid. `u` is the
// c_up=2 carrier (.xy); `rgb` is x_hr (.xyz). Output .x = alpha (4-ch tensor).
// weights.weights = 18 vec4 (split-packed); weights.bias = 1.
export class UpFinalWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, u: Tensor, rgb: Tensor, w: Conv2DWeights) {
    super(backend);
    this.shader = backend.dtype === "f16" ? upFinalF16Src : upFinalF32Src;

    this.output  = backend.tensor(u.h, u.w, 4);
    this.inputs  = [u, rgb];
    this.weights = [
      backend.upload(toUploadView(w.weights)),
      backend.upload(padToVec4(w.bias)),
    ];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([u.h, u.w, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(u.w / 8), Math.ceil(u.h / 8), 1];
  }
}
