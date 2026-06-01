import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import candF32Src from "~/model/backends/webgpu/shaders/cand_update_fused.wgsl";
import candF16Src from "~/model/backends/webgpu/shaders/cand_update_fused_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// ConvGRU candidate + state update + output, fused (production config c_up=2,
// recurrent=1). `gatesOut` is gates_fused's (z, r) output. weights.weights =
// 9 vec4 (.xy = b_w, rh_w); weights.bias = (cand_bias); gamma = recurrent scale.
// Output is a 4-channel tensor packing (a, b_out, 0, 0).
export class CandUpdateFusedWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(
    backend: WebGPUBackend,
    uIn: Tensor,
    hPrev: Tensor,
    gatesOut: Tensor,
    w: Conv2DWeights,
    gamma: ArrayLike<number>,
  ) {
    super(backend);
    this.shader = backend.dtype === "f16" ? candF16Src : candF32Src;

    this.output  = backend.tensor(uIn.h, uIn.w, 4);
    this.inputs  = [uIn, hPrev, gatesOut];
    this.weights = [
      backend.upload(toUploadView(w.weights)),
      backend.upload(padToVec4(w.bias)),
      backend.upload(padToVec4(gamma)),
    ];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([uIn.h, uIn.w, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(uIn.w / 8), Math.ceil(uIn.h / 8), 1];
  }
}
