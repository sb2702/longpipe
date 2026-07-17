import type { Tensor, ReframeStateParams, MLBuffer } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import { WebGPUTensor, WebGPUOp } from "~/model/backends/webgpu/base_webgpu_op.ts";
import f32Src from "~/model/backends/webgpu/shaders/reframe_state.wgsl";
import f16Src from "~/model/backends/webgpu/shaders/reframe_state_f16.wgsl";

// Auto-reframe camera state: (boxes, prev state, cmd) → new state (1×1×4).
// One thread. The renderer threads `prev` across frames with copyTensor — the
// same carrier pattern as the flow stabilizer, so no readback.
// Binding order (inputs → uniforms → output): 0 boxes, 1 prev, 2 cmd, 3 params, 4 out.
export class ReframeStateWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly weights: MLBuffer[] = [];
  readonly output: WebGPUTensor;
  protected dispatch: [number, number, number] = [1, 1, 1];
  shader: string;

  constructor(backend: WebGPUBackend, boxes: Tensor, prev: Tensor, cmd: Tensor, params: ReframeStateParams) {
    super(backend);
    this.shader = backend.dtype === "f16" ? f16Src : f32Src;
    this.output = backend.tensor(1, 1, 8);   // [0] view rect, [1] subject memory
    this.inputs = [boxes, prev, cmd];

    this.createUniform("params", "Params");
    const ab = new ArrayBuffer(32);
    new Uint32Array(ab, 0, 1)[0] = boxes.w * boxes.h;
    const f = new Float32Array(ab, 4, 7);
    f[0] = params.zoom; f[1] = params.gravity; f[2] = params.margin;
    f[3] = params.deadband; f[4] = params.ease; f[5] = params.aspect;
    this.setUniform("params", new Uint32Array(ab));

    this.defaultSetup();
  }
}
