import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import gatesF32Src from "~/model/backends/webgpu/shaders/gates_fused.wgsl";
import gatesF16Src from "~/model/backends/webgpu/shaders/gates_fused_f16.wgsl";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// ConvGRU z + r gates, fused (production config c_up=2, recurrent=1).
// `uIn` is the c_up=2 feature (.x=a, .y=b); `hPrev` is the 1-channel hidden.
// weights.weights = 9 vec4 (36 floats); weights.bias = (z_bias, r_bias).
// Output is a 4-channel tensor packing (z, r, 0, 0).
export class GatesFusedWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, uIn: Tensor, hPrev: Tensor, w: Conv2DWeights) {
    super(backend);
    this.shader = backend.dtype === "f16" ? gatesF16Src : gatesF32Src;

    this.output  = backend.tensor(uIn.h, uIn.w, 4);
    this.inputs  = [uIn, hPrev];
    this.weights = [
      backend.upload(toUploadView(w.weights)),
      backend.upload(padToVec4(w.bias)),
    ];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([uIn.h, uIn.w, 0, 0]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(uIn.w / 8), Math.ceil(uIn.h / 8), 1];
  }
}
