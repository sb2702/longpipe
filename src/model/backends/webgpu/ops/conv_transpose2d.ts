import type { Tensor, ConvTranspose2dParams } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/conv_transpose2d.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/conv_transpose2d_f16.wgsl";
import { convTransposeOutSize } from "~/model/backends/webgpu/ops/conv_utils.ts";
import { toUploadView } from "~/utils/weights.ts";

// Gather-form transposed conv. Weight buffer is uploaded raw — identical flat
// layout to Conv2d (mat4x4[z][o][i], M[in_sub][out_sub] = W(in, out, ky, kx)).
export class ConvTranspose2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: import("~/model/backend").MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, w: Conv2DWeights, params: ConvTranspose2dParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    const outH = convTransposeOutSize(input.h, params.kernel, params.stride, params.padding);
    const outW = convTransposeOutSize(input.w, params.kernel, params.stride, params.padding);
    const inGroups  = input.c / 4;
    const outGroups = params.outChannels / 4;

    this.output  = backend.tensor(outH, outW, params.outChannels);
    this.inputs  = [input];
    this.weights = [backend.upload(toUploadView(w.weights)), backend.upload(toUploadView(w.bias))];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, outH, outW,
      inGroups, outGroups,
      params.kernel, params.kernel,
      params.stride, params.padding, params.padding,
      params.activation === "relu6" ? 1 : params.activation === "relu" ? 2 : params.activation === "leaky" ? 3 : 0,
    ]));

    this.defaultSetup();

    // One thread per (ox, oy, out-group); workgroup z = 1, so dispatch z = outGroups.
    this.dispatch = [Math.ceil(outW / 8), Math.ceil(outH / 8), outGroups];
  }
}
