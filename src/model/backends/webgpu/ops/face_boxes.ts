import type { Tensor, FaceBoxesParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/face_boxes.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/face_boxes_f16.wgsl";

// Multi-face box decode: 5-keypoint heatmaps → 1×K×4 boxes (cx, cy, halfSide,
// score in frame fractions), K = params.maxFaces. Local-max candidates →
// eye-pair hypotheses → geometric scoring → greedy NMS, all in one tiny
// dispatch (single 5-thread workgroup — the grid is ≤ ~48×28).
// Binding order (inputs → uniforms → output): 0 heatmaps, 1 params, 2 output.
export class FaceBoxesWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number] = [1, 1, 1];
  shader: string;

  constructor(backend: WebGPUBackend, heatmaps: Tensor, params: FaceBoxesParams) {
    super(backend);
    if (heatmaps.c !== 8)
      throw new Error(`FaceBoxesFromHeatmaps: expected 8-ch heatmaps (5 real), got ${heatmaps.c}`);
    if (params.maxFaces < 1 || params.maxFaces > 6)
      throw new Error(`FaceBoxesFromHeatmaps: maxFaces must be 1..6, got ${params.maxFaces}`);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;

    this.output = backend.tensor(1, params.maxFaces, 4);
    this.inputs = [heatmaps];

    this.createUniform("params", "Params");
    // { h, w: u32, win: i32, thresh, box_scale: f32, max_faces: u32, tol: f32 }
    // 7 × 4B = 28 → padded to the 16B uniform alignment.
    const ab = new ArrayBuffer(32);
    const u = new Uint32Array(ab);
    const i = new Int32Array(ab);
    const f = new Float32Array(ab);
    u[0] = heatmaps.h;
    u[1] = heatmaps.w;
    i[2] = params.win;
    f[3] = params.thresh;
    f[4] = params.boxScale;
    u[5] = params.maxFaces;
    f[6] = params.tol;
    this.setUniform("params", u);

    this.defaultSetup();
  }
}
