import type { ImageSource, InputOp, Dtype } from "~/model/backend.ts";
import type { WebGPUBackend } from "~/model/backends/webgpu/index.ts";
import type { WebGPUTensor } from "~/model/backends/webgpu/base_webgpu_op.ts";
import input2dF32Src        from "~/model/backends/webgpu/shaders/input_2d.wgsl";
import input2dF16Src        from "~/model/backends/webgpu/shaders/input_2d_f16.wgsl";
import inputExternalF32Src  from "~/model/backends/webgpu/shaders/input_external.wgsl";
import inputExternalF16Src  from "~/model/backends/webgpu/shaders/input_external_f16.wgsl";
import { orientationUvMatrix, sameOrientation, type Orientation } from "~/model/orientation.ts";

const IS_VIDEO_FRAME = (s: ImageSource): s is VideoFrame =>
  typeof VideoFrame !== "undefined" && s instanceof VideoFrame;

// Lazily builds two compute pipelines — one for static 2D sources (RGBA8
// staging texture, fed by copyExternalImageToTexture) and one for VideoFrames
// (zero-copy via importExternalTexture). Pipeline + sampler + uniform buffer
// + output Tensor are stable; bind groups and the import / copy happen in
// run().
export class InputWebGPU implements InputOp {
  readonly output: WebGPUTensor;

  private readonly backend: WebGPUBackend;
  private readonly device: GPUDevice;
  private readonly dtype: Dtype;
  private readonly sampler: GPUSampler;
  private readonly uniformBuffer: GPUBuffer;
  private readonly dispatch: [number, number, number];

  // Lazily-compiled pipelines and the staging texture for the 2D path.
  private pipeline2d:        GPUComputePipeline | null = null;
  private pipelineExternal:  GPUComputePipeline | null = null;
  private stagingTex:        GPUTexture | null = null;
  private stagingW = 0;
  private stagingH = 0;

  private source: ImageSource | null = null;
  private readonly outW: number;
  private readonly outH: number;
  // Orientation currently in the uniform buffer (null = identity).
  private orientation: Orientation | null = null;

  constructor(backend: WebGPUBackend, h: number, w: number) {
    this.backend = backend;
    this.device = backend.device;
    this.dtype  = backend.dtype;
    this.output = backend.tensor(h, w, 4);

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 48, // out_w u32 + out_h u32 + 8 padding + uv_x vec4 + uv_y vec4
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.outW = w;
    this.outH = h;
    this.writeParams(null);

    this.dispatch = [Math.ceil(w / 8), Math.ceil(h / 8), 1];

    // Pre-compile the 2D pipeline since it's the common static path; the
    // external-texture pipeline is built on demand the first time a
    // VideoFrame source is set (avoids paying the cost when callers only ever
    // pass ImageBitmaps).
    this.pipeline2d = this.buildPipeline(
      this.dtype === "f16" ? input2dF16Src : input2dF32Src,
    );
  }

  // `orientation`: the display transform still to apply to `src` (rotation /
  // flip the browser's upload path won't apply — see model/orientation.ts).
  // Applied in the sampling shader; omitted = identity.
  setSource(src: ImageSource, orientation?: Orientation | null): void {
    this.source = src;
    if (!sameOrientation(orientation, this.orientation)) this.writeParams(orientation ?? null);
  }

  // The uniform write is queue-timeline, ahead of the (batched) dispatch that
  // reads it, and only happens when the orientation changes.
  private writeParams(o: Orientation | null): void {
    const ab = new ArrayBuffer(48);
    new Uint32Array(ab, 0, 2).set([this.outW, this.outH]);
    const m = orientationUvMatrix(o);
    new Float32Array(ab, 16, 8).set([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ab);
    this.orientation = o;
  }

  run(): void {
    if (!this.source) throw new Error("InputWebGPU.run() called before setSource()");

    if (IS_VIDEO_FRAME(this.source)) {
      if (!this.externalUnsupported) {
        try {
          this.runExternal(this.source);
          return;
        } catch (err) {
          // Some WebGPU implementations (e.g. Firefox, where external textures
          // are behind a pref) reject importExternalTexture. Fall back to the
          // copy path for the rest of this op's life.
          this.externalUnsupported = true;
          console.warn("[longpipe] importExternalTexture unavailable, using copyExternalImageToTexture:", (err as Error)?.message ?? err);
        }
      }
      this.run2d(this.source);
    } else {
      this.run2d(this.source);
    }
  }

  private externalUnsupported = false;

  private run2d(src: ImageBitmap | VideoFrame): void {
    const w = IS_VIDEO_FRAME(src) ? src.displayWidth : src.width;
    const h = IS_VIDEO_FRAME(src) ? src.displayHeight : src.height;
    this.ensureStagingTexture(w, h);
    // Queue-timeline copy: submit any batched work first so it stays ordered
    // (earlier recorded passes may still read the staging texture).
    this.backend.flush();
    this.device.queue.copyExternalImageToTexture(
      { source: src, flipY: false },
      { texture: this.stagingTex! },
      [w, h],
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline2d!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.stagingTex!.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
        { binding: 3, resource: { buffer: this.output.buffer } },
      ],
    });
    this.dispatchOnce(this.pipeline2d!, bindGroup);
  }

  private runExternal(src: VideoFrame): void {
    if (typeof this.device.importExternalTexture !== "function") throw new Error("importExternalTexture not supported");
    if (!this.pipelineExternal) {
      this.pipelineExternal = this.buildPipeline(
        this.dtype === "f16" ? inputExternalF16Src : inputExternalF32Src,
      );
    }
    const externalTex = this.device.importExternalTexture({ source: src });
    const bindGroup = this.device.createBindGroup({
      layout: this.pipelineExternal.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTex },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
        { binding: 3, resource: { buffer: this.output.buffer } },
      ],
    });
    this.dispatchOnce(this.pipelineExternal, bindGroup);
  }

  private ensureStagingTexture(w: number, h: number): void {
    if (this.stagingTex && this.stagingW === w && this.stagingH === h) return;
    this.stagingTex?.destroy();
    this.stagingTex = this.device.createTexture({
      size: [w, h, 1],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.stagingW = w;
    this.stagingH = h;
  }

  private buildPipeline(code: string): GPUComputePipeline {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private dispatchOnce(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup): void {
    const shared = this.backend.computePass();   // inside backend.batch()
    if (shared) {
      shared.setPipeline(pipeline);
      shared.setBindGroup(0, bindGroup);
      shared.dispatchWorkgroups(...this.dispatch);
      return;
    }
    const enc  = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...this.dispatch);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
