import type { Tensor, MLBuffer, DownAdapterParams } from "~/model/backend.ts";
import type { Conv2DWeights } from "~/model/weights.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import downAdapterF32Src from "~/model/backends/webgpu/shaders/down_adapter.wgsl";
import downAdapterF16Src from "~/model/backends/webgpu/shaders/down_adapter_f16.wgsl";
import { convOutSize, resolvePad } from "~/model/backends/webgpu/ops/conv_utils.ts";
import { toUploadView, padToVec4 } from "~/utils/weights.ts";

// Fused stride-N 3×3 conv (4→4) + relu + 1×1 adapter (4→3). Symmetric pad 1.
// `input` is 4-channel; output is the base input vec4(adapter.xyz, 0).
// downWeights = 9 mat4x4 (3×3 4→4); adaptWeights = 1 mat4x4 (1×1, padded 4→3).
export class DownAdapterWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, input: Tensor, downW: Conv2DWeights, adaptW: Conv2DWeights, params: DownAdapterParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? downAdapterF16Src : downAdapterF32Src;

    const outH    = convOutSize(input.h, 3, params.stride, 1);
    const outW    = convOutSize(input.w, 3, params.stride, 1);
    const padTop  = resolvePad(1, input.h, outH, 3, params.stride);
    const padLeft = resolvePad(1, input.w, outW, 3, params.stride);

    this.output  = backend.tensor(outH, outW, 4);
    this.inputs  = [input];
    this.weights = [
      backend.upload(toUploadView(downW.weights)),
      backend.upload(padToVec4(downW.bias)),
      backend.upload(toUploadView(adaptW.weights)),
      backend.upload(padToVec4(adaptW.bias)),
    ];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([
      input.h, input.w, outH, outW, params.stride, padTop, padLeft, 0,
    ]));

    this.defaultSetup();

    this.dispatch = [Math.ceil(outW / 8), Math.ceil(outH / 8), 1];
  }
}
