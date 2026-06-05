import type { Tensor, WarpParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/warp.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/warp_f16.wgsl";

// Bilinear gather-warp. Binding order (inputs → uniforms → output):
// 0 source, 1 flow, 2 params, 3 output.
export class WarpWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number];
  shader: string;

  constructor(backend: WebGPUBackend, source: Tensor, flow: Tensor, params: WarpParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    this.output = backend.tensor(source.h, source.w, source.c);
    this.inputs = [source, flow];

    this.createUniform("params", "Params");
    const u = new Uint32Array(3);              // { h: u32, w: u32, flow_scale: f32 }
    u[0] = source.h;
    u[1] = source.w;
    new Float32Array(u.buffer)[2] = params.flowScale;
    this.setUniform("params", u);

    this.defaultSetup();
    this.dispatch = [Math.ceil(source.w / 8), Math.ceil(source.h / 8), 1];
  }
}
