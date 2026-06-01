import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import catConv6to2F32Src from "~/model/backends/webgpu/shaders/cat_conv_6to2.wgsl";
import catConv6to2F16Src from "~/model/backends/webgpu/shaders/cat_conv_6to2_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// Fused concat(u, d) + 6→2 conv 3×3 + relu (E up1_combine). `u` is the c_up=2
// carrier (.xy); `d` is the c_high=4 skip (full vec4); same resolution. Output
// is the c_up=2 carrier (.xy = 2 native channels, .zw = 0).
// weights.weights = 9 * 2 mat3x2 (6 floats each); weights.bias = 2.
export class CatConv6to2WebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, u: Tensor, d: Tensor, w: Conv2DWeights) {
    super(backend);
    this.shader = backend.dtype === "f16" ? catConv6to2F16Src : catConv6to2F32Src;

    this.output  = backend.tensor(u.h, u.w, 4);
    this.inputs  = [u, d];
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
