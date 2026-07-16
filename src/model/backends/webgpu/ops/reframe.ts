import type { Tensor, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/reframe.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/reframe_f16.wgsl";

// Apply the view rect to a tensor (same shape in/out). Identity while the rect
// is uninitialised (size ≤ 0), so wiring it in costs nothing until a face is found.
// Binding order (inputs → uniforms → output): 0 src, 1 rect, 2 params, 3 out.
export class ReframeWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, src: Tensor, rect: Tensor) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;
    this.output = backend.tensor(src.h, src.w, 4);
    this.inputs = [src, rect];

    this.createUniform("params", "Params");
    this.setUniform("params", new Uint32Array([src.h, src.w]));

    this.defaultSetup();
    this.dispatch = [Math.ceil(src.w / 8), Math.ceil(src.h / 8), 1];
  }
}
