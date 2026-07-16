import type { Backend, Dtype, DataView_, RenderTarget } from "~/model/backend.ts";
import type { WebGPUTensor, WebGPUMLBuffer } from "~/model/backends/webgpu/base_webgpu_op.ts";
import { float32ArrayToHalf, halfArrayToFloat32 } from "~/utils/fp16.ts";
import { Conv2DWebGPU } from "~/model/backends/webgpu/ops/conv2d.ts";
import { ConvTranspose2DWebGPU } from "~/model/backends/webgpu/ops/conv_transpose2d.ts";
import { DepthwiseConv2DWebGPU } from "~/model/backends/webgpu/ops/depthwise_conv2d.ts";
import { AddWebGPU } from "~/model/backends/webgpu/ops/add.ts";
import { SigmoidWebGPU } from "~/model/backends/webgpu/ops/sigmoid.ts";
import { TanhWebGPU } from "~/model/backends/webgpu/ops/tanh.ts";
import { ElementwiseMulWebGPU } from "~/model/backends/webgpu/ops/elementwise_mul.ts";
import { WarpWebGPU } from "~/model/backends/webgpu/ops/warp.ts";
import { FaceBoxWebGPU } from "~/model/backends/webgpu/ops/face_box.ts";
import { FaceBoxesWebGPU } from "~/model/backends/webgpu/ops/face_boxes.ts";
import { ReframeStateWebGPU } from "~/model/backends/webgpu/ops/reframe_state.ts";
import { ReframeWebGPU } from "~/model/backends/webgpu/ops/reframe.ts";
import { CropResampleWebGPU } from "~/model/backends/webgpu/ops/crop_resample.ts";
import { LandmarkOverlayWebGPU } from "~/model/backends/webgpu/ops/landmark_overlay.ts";
import { FaceTouchupWebGPU, FaceTouchupStageWebGPU } from "~/model/backends/webgpu/ops/face_touchup.ts";
import { StabilizeWebGPU } from "~/model/backends/webgpu/ops/stabilize.ts";
import { BilinearUpsampleWebGPU } from "~/model/backends/webgpu/ops/bilinear_upsample.ts";
import { CropWebGPU } from "~/model/backends/webgpu/ops/crop.ts";
import { BicubicUpsampleWebGPU  } from "~/model/backends/webgpu/ops/bicubic_upsample.ts";
import { ChannelConcatWebGPU } from "~/model/backends/webgpu/ops/channel_concat.ts";
import { Conv2dAddWebGPU } from "~/model/backends/webgpu/ops/conv2d_add.ts";
import { ProjResidualWebGPU } from "~/model/backends/webgpu/ops/proj_residual.ts";
import { ConcatConv2dWebGPU } from "~/model/backends/webgpu/ops/concat_conv2d.ts";
import { GatesFusedWebGPU } from "~/model/backends/webgpu/ops/gates_fused.ts";
import { CandUpdateFusedWebGPU } from "~/model/backends/webgpu/ops/cand_update_fused.ts";
import { ConvExpandWebGPU } from "~/model/backends/webgpu/ops/conv_expand.ts";
import { CatConv6to2WebGPU } from "~/model/backends/webgpu/ops/cat_conv_6to2.ts";
import { DownAdapterWebGPU } from "~/model/backends/webgpu/ops/down_adapter.ts";
import { UpFinalWebGPU } from "~/model/backends/webgpu/ops/up_final.ts";
import { UpFinalSkipWebGPU } from "~/model/backends/webgpu/ops/up_final_skip.ts";
import { UpsampleConcatWebGPU } from "~/model/backends/webgpu/ops/upsample_concat.ts";
import { UpsampleConv1x1WebGPU } from "~/model/backends/webgpu/ops/upsample_conv1x1.ts";
import { UpsampleSigmoidWebGPU } from "~/model/backends/webgpu/ops/upsample_sigmoid.ts";
import { CompositeSolidWebGPU } from "~/model/backends/webgpu/ops/composite_solid.ts";
import { CompositeImageWebGPU } from "~/model/backends/webgpu/ops/composite_image.ts";
import { CompositeImageBilinearWebGPU } from "~/model/backends/webgpu/ops/composite_image_bilinear.ts";
import { CompositePassthroughWebGPU } from "~/model/backends/webgpu/ops/composite_passthrough.ts";
import { InputWebGPU } from "~/model/backends/webgpu/ops/input.ts";

const STORAGE = navigator.gpu ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST : 0;

export interface WebGPUBackendOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  device?: GPUDevice;
  // Defaults to 'f32'. 'f16' requires the device to expose the `shader-f16`
  // feature; create() throws otherwise.
  dtype?: Dtype;
}

export class WebGPUBackend implements Backend {
  readonly ops: Backend["ops"];
  readonly presenters: Backend["presenters"];
  readonly canvasContext: GPUCanvasContext;          // the 'main' target's context
  readonly canvasFormat: GPUTextureFormat;
  readonly bytesPerElement: 2 | 4;

  // Configured GPUCanvasContexts keyed by render target. Seeded with 'main'
  // (the create() canvas); attachCanvas() adds others (e.g. 'preview'). One
  // device drives them all, so render passes to different canvases see the
  // same device buffers — no cross-context sharing problem.
  private readonly contexts = new Map<RenderTarget, GPUCanvasContext>();

  private constructor(
    readonly device: GPUDevice,
    readonly canvas: HTMLCanvasElement | OffscreenCanvas,
    readonly dtype: Dtype,
  ) {
    this.bytesPerElement = dtype === "f16" ? 2 : 4;

    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("Failed to get WebGPU context from canvas");
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.configureContext(ctx);
    this.canvasContext = ctx;
    this.contexts.set("main", ctx);

    this.ops = {
      Conv2d:           (input, weights, params)        => new Conv2DWebGPU(this, input, weights, params),
      ConvTranspose2d:  (input, weights, params)        => new ConvTranspose2DWebGPU(this, input, weights, params),
      DepthwiseConv2d:  (input, weights, params)        => new DepthwiseConv2DWebGPU(this, input, weights, params),
      Add:              (a, b)                          => new AddWebGPU(this, a, b),
      Sigmoid:          (input)                         => new SigmoidWebGPU(this, input),
      Tanh:             (input)                         => new TanhWebGPU(this, input),
      ElementwiseMul:   (a, b)                          => new ElementwiseMulWebGPU(this, a, b),
      Warp:             (source, flow, params)          => new WarpWebGPU(this, source, flow, params),
      FaceBoxFromHeatmaps: (heatmaps, params)            => new FaceBoxWebGPU(this, heatmaps, params),
      FaceBoxesFromHeatmaps: (heatmaps, params)          => new FaceBoxesWebGPU(this, heatmaps, params),
      ReframeState: (boxes, prev, cmd, params)           => new ReframeStateWebGPU(this, boxes, prev, cmd, params),
      Reframe: (src, rect)                               => new ReframeWebGPU(this, src, rect),
      CropResample:     (frame, box, params)             => new CropResampleWebGPU(this, frame, box, params),
      FaceTouchupStage: (frame, landmarks, box, topo, params) => new FaceTouchupStageWebGPU(this, frame, landmarks, box, topo, params),
      Stabilize:        (flow, pred, ref, envPrev, params) => new StabilizeWebGPU(this, flow, pred, ref, envPrev, params),
      BilinearUpsample: (input, params)                 => new BilinearUpsampleWebGPU(this, input, params),
      Crop:             (input, params)                 => new CropWebGPU(this, input, params),
      BicubicUpsample:  (input, params)                 => new BicubicUpsampleWebGPU(this, input, params),
      ChannelConcat:    (a, b)                          => new ChannelConcatWebGPU(this, a, b),
      Conv2dAdd:        (input, skip, weights, params)  => new Conv2dAddWebGPU(this, input, skip, weights, params),
      ProjResidual:     (input, skip, weights, params)  => new ProjResidualWebGPU(this, input, skip, weights, params),
      ConcatConv2d:     (a, b, weights, params)         => new ConcatConv2dWebGPU(this, a, b, weights, params),
      GatesFused:       (uIn, hPrev, weights)           => new GatesFusedWebGPU(this, uIn, hPrev, weights),
      CandUpdateFused:  (uIn, hPrev, gatesOut, w, gamma) => new CandUpdateFusedWebGPU(this, uIn, hPrev, gatesOut, w, gamma),
      ConvExpand:       (input, weights)                 => new ConvExpandWebGPU(this, input, weights),
      CatConv6to2:      (u, d, weights)                  => new CatConv6to2WebGPU(this, u, d, weights),
      DownAdapter:      (input, downW, adaptW, params)   => new DownAdapterWebGPU(this, input, downW, adaptW, params),
      UpFinal:          (u, rgb, weights)               => new UpFinalWebGPU(this, u, rgb, weights),
      UpFinalSkip:      (u, dFull, rgb, weights)        => new UpFinalSkipWebGPU(this, u, dFull, rgb, weights),
      UpsampleConcat:   (a, b, params)                  => new UpsampleConcatWebGPU(this, a, b, params),
      UpsampleConv1x1:  (input, weights, params)        => new UpsampleConv1x1WebGPU(this, input, weights, params),
      UpsampleSigmoid:  (input, params)                 => new UpsampleSigmoidWebGPU(this, input, params),
      Input:            (h, w)                          => new InputWebGPU(this, h, w),
    };

    this.presenters = {
      // The op needs the per-frame swapchain texture of its target canvas
      // before each draw — wrap it so callers don't have to think about
      // presentation lifecycle. `target` defaults to 'main'.
      CompositeSolid: (image, alpha, bgColor, target = "main") => {
        const op = new CompositeSolidWebGPU(this, image, alpha, bgColor);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
      CompositeImage: (image, alpha, bg, target = "main") => {
        const op = new CompositeImageWebGPU(this, image, alpha, bg);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
      CompositeImageBilinear: (image, alpha, bg, target = "main") => {
        const op = new CompositeImageBilinearWebGPU(this, image, alpha, bg);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
      CompositePassthrough: (image, target = "main") => {
        const op = new CompositePassthroughWebGPU(this, image);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
      LandmarkOverlay: (image, landmarks, box, params, target = "main") => {
        const op = new LandmarkOverlayWebGPU(this, image, landmarks, box, params);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
      FaceTouchup: (frame, landmarks, box, topo, params, target = "main") => {
        const op = new FaceTouchupWebGPU(this, frame, landmarks, box, topo, params);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture(target));
            op.run();
          },
        };
      },
    };
  }

  // Configure a GPUCanvasContext for this backend's device + format. Shared by
  // the constructor (main) and attachCanvas (preview).
  private configureContext(ctx: GPUCanvasContext): void {
    ctx.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: "premultiplied",
    });
  }

  // Register a second output canvas (see Backend.attachCanvas). 'main' is
  // reserved for the create() canvas.
  attachCanvas(name: RenderTarget, canvas: HTMLCanvasElement | OffscreenCanvas): void {
    if (name === "main") throw new Error("attachCanvas: 'main' is reserved for the create() canvas");
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error(`attachCanvas: failed to get WebGPU context for target '${name}'`);
    this.configureContext(ctx);
    this.contexts.set(name, ctx);
  }

  static async isAvailable(): Promise<boolean> {
    if (!navigator.gpu) return false;
    return (await navigator.gpu.requestAdapter()) !== null;
  }

  static async hasF16Support(): Promise<boolean> {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? adapter.features.has("shader-f16") : false;
  }

  static async create(opts: WebGPUBackendOptions): Promise<WebGPUBackend> {
    const dtype: Dtype = opts.dtype ?? "f32";
    let device = opts.device;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("WebGPU adapter not available");
      if (dtype === "f16" && !adapter.features.has("shader-f16"))
        throw new Error("WebGPU dtype='f16' requested but adapter lacks `shader-f16` feature");
      device = await adapter.requestDevice({
        requiredFeatures: dtype === "f16" ? (["shader-f16"] as GPUFeatureName[]) : [],
      });
    } else if (dtype === "f16" && !device.features.has("shader-f16")) {
      throw new Error("WebGPU dtype='f16' requested but supplied device lacks `shader-f16`");
    }
    return new WebGPUBackend(device, opts.canvas, dtype);
  }

  // The swapchain texture for the current frame of `target`'s canvas (default
  // 'main'). Must be called inside the same task that submits the render
  // commands — texture is invalidated after the next browser paint.
  getCurrentDisplayTexture(target: RenderTarget = "main"): GPUTexture {
    const ctx = this.contexts.get(target);
    if (!ctx) throw new Error(`getCurrentDisplayTexture: no canvas attached for target '${target}'`);
    return ctx.getCurrentTexture();
  }

  tensor(h: number, w: number, c: number, data?: DataView_): WebGPUTensor {
    const elements = h * w * c;
    const bytes = elements * this.bytesPerElement;
    const buf = this.device.createBuffer({
      size: bytes,
      usage: STORAGE,
      mappedAtCreation: data !== undefined,
    });
    if (data !== undefined) {
      const range = buf.getMappedRange();
      this.writeView(range, data);
      buf.unmap();
    }
    return { h, w, c, buffer: buf };
  }

  upload(data: DataView_): WebGPUMLBuffer {
    const elements = data.length;
    const bytes = elements * this.bytesPerElement;
    const buf = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    this.writeView(buf.getMappedRange(), data);
    buf.unmap();
    return { buffer: buf };
  }

  // Write `data` (Float32 or Uint16-fp16-bits) into the mapped buffer range,
  // converting to match the backend's dtype if necessary.
  private writeView(range: ArrayBuffer, data: DataView_): void {
    const wantHalf = this.dtype === "f16";
    const isHalf = data instanceof Uint16Array;
    if (wantHalf === isHalf) {
      // Storage matches source — copy bytes directly.
      if (wantHalf) new Uint16Array(range).set(data as Uint16Array);
      else          new Float32Array(range).set(data as Float32Array);
      return;
    }
    if (wantHalf) {
      // f32 input → f16 storage
      new Uint16Array(range).set(float32ArrayToHalf(data as Float32Array));
    } else {
      // f16 input → f32 storage
      new Float32Array(range).set(halfArrayToFloat32(data as Uint16Array));
    }
  }


  async readback(tensor: WebGPUTensor): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      size: tensor.buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(tensor.buffer, 0, staging, 0, tensor.buffer.size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const result = this.dtype === "f16"
      ? halfArrayToFloat32(new Uint16Array(mapped.slice(0)))
      : new Float32Array(mapped.slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  // GPU-resident buffer→buffer copy (no CPU round-trip). Both tensor buffers
  // carry COPY_SRC|COPY_DST (see STORAGE usage), so this is a plain enqueued
  // DMA on the device queue. src/dst must be the same byte size.
  copyTensor(src: WebGPUTensor, dst: WebGPUTensor): void {
    if (src.buffer.size !== dst.buffer.size)
      throw new Error(`copyTensor: size mismatch (src ${src.buffer.size} vs dst ${dst.buffer.size})`);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src.buffer, 0, dst.buffer, 0, src.buffer.size);
    this.device.queue.submit([enc.finish()]);
  }

  // Awaits all in-flight queue work. Used for benchmarking when we want
  // a sync barrier without paying the readback bandwidth cost.
  async sync(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    this.device.destroy();
  }
}
