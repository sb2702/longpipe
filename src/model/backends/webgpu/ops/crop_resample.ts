import type { Tensor, CropResampleParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/crop_resample.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/crop_resample_f16.wgsl";

// Box-driven square crop + bilinear resample + ((rgb - mean)/std) normalize.
// Binding order (inputs → uniforms → output): 0 frame, 1 box, 2 params, 3 output.
export class CropResampleWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, frame: Tensor, box: Tensor, params: CropResampleParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    this.output = backend.tensor(params.outH, params.outW, 4);
    this.inputs = [frame, box];

    const slot = params.slot ?? 0;
    if (slot >= box.w * box.h)
      throw new Error(`CropResample: slot ${slot} out of range for a ${box.h}×${box.w} box tensor`);

    this.createUniform("params", "Params");
    // 5×u32 (+3 pad — vec4 members need 16B alignment) + vec4 mean + vec4 std.
    const ab = new ArrayBuffer(64);
    const u = new Uint32Array(ab, 0, 5);
    u[0] = frame.h; u[1] = frame.w; u[2] = params.outH; u[3] = params.outW; u[4] = slot;
    new Float32Array(ab, 32, 4).set([...params.mean, 0]);
    new Float32Array(ab, 48, 4).set([...params.std, 1]);
    this.setUniform("params", new Uint32Array(ab));

    this.defaultSetup();
    this.dispatch = [Math.ceil(params.outW / 8), Math.ceil(params.outH / 8), 1];
  }
}
