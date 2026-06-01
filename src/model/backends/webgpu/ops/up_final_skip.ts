import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import upFinalSkipF32Src from "~/model/backends/webgpu/shaders/up_final_skip.wgsl";
import upFinalSkipF16Src from "~/model/backends/webgpu/shaders/up_final_skip_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// C/D alpha head: fused concat(u, d_full, rgb) → conv 3×3 9→1 → sigmoid. `u` is
// the c_up=2 carrier (.xy); `dFull` is the c_high=4 full-res skip (full vec4);
// `rgb` is x_hr (.xyz). Output .x = alpha (4-ch tensor).
// weights.weights = 27 vec4 (split-packed); weights.bias = 1.
export class UpFinalSkipWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, u: Tensor, dFull: Tensor, rgb: Tensor, w: Conv2DWeights) {
    super(backend);
    this.shader = backend.dtype === "f16" ? upFinalSkipF16Src : upFinalSkipF32Src;

    this.output  = backend.tensor(u.h, u.w, 4);
    this.inputs  = [u, dFull, rgb];
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
