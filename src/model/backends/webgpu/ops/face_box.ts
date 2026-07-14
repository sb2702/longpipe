import type { Tensor, FaceBoxParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/face_box.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/face_box_f16.wgsl";

// Face-box decode: 5-keypoint heatmaps → 1×1×4 (cx, cy, halfSide, score) in
// frame fractions. Soft-argmax per channel + hull → pixel-square box, all in
// one tiny dispatch (single 5-thread workgroup — the grid is ≤ ~48×28).
// Binding order (inputs → uniforms → output): 0 heatmaps, 1 params, 2 output.
export class FaceBoxWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number] = [1, 1, 1];
  shader: string;

  constructor(backend: WebGPUBackend, heatmaps: Tensor, params: FaceBoxParams) {
    super(backend);
    if (heatmaps.c !== 8)
      throw new Error(`FaceBoxFromHeatmaps: expected 8-ch heatmaps (5 real), got ${heatmaps.c}`);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    this.output = backend.tensor(1, 1, 4);
    this.inputs = [heatmaps];

    this.createUniform("params", "Params");
    const u = new Uint32Array(5);   // { h, w: u32, win: i32, thresh, box_scale: f32 }
    u[0] = heatmaps.h;
    u[1] = heatmaps.w;
    new Int32Array(u.buffer)[2] = params.win;
    const f = new Float32Array(u.buffer);
    f[3] = params.thresh;
    f[4] = params.boxScale;
    this.setUniform("params", u);

    this.defaultSetup();
  }
}
