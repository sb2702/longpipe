import type { ImageSource, InputOp, Dtype } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import type { WebGPUTensor } from "~/model/backends/webgpu/base_webgpu_op";
import input2dF32Src        from "~/model/backends/webgpu/shaders/input_2d.wgsl";
import input2dF16Src        from "~/model/backends/webgpu/shaders/input_2d_f16.wgsl";
import inputExternalF32Src  from "~/model/backends/webgpu/shaders/input_external.wgsl";
import inputExternalF16Src  from "~/model/backends/webgpu/shaders/input_external_f16.wgsl";

const IS_VIDEO_FRAME = (s: ImageSource): s is VideoFrame =>
  typeof VideoFrame !== "undefined" && s instanceof VideoFrame;

// Lazily builds two compute pipelines — one for static 2D sources (RGBA8
// staging texture, fed by copyExternalImageToTexture) and one for VideoFrames
// (zero-copy via importExternalTexture). Pipeline + sampler + uniform buffer
// + output Tensor are stable; bind groups and the import / copy happen in
// run().
export class InputWebGPU implements InputOp {
  readonly output: WebGPUTensor;

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

  constructor(backend: WebGPUBackend, h: number, w: number) {
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
      size: 16, // out_w u32 + out_h u32 + 8 padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const ab = new ArrayBuffer(16);
    new Uint32Array(ab, 0, 2).set([w, h]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ab);

    this.dispatch = [Math.ceil(w / 8), Math.ceil(h / 8), 1];

    // Pre-compile the 2D pipeline since it's the common static path; the
    // external-texture pipeline is built on demand the first time a
    // VideoFrame source is set (avoids paying the cost when callers only ever
    // pass ImageBitmaps).
    this.pipeline2d = this.buildPipeline(
      this.dtype === "f16" ? input2dF16Src : input2dF32Src,
    );
  }

  setSource(src: ImageSource): void {
    this.source = src;
  }

  run(): void {
    if (!this.source) throw new Error("InputWebGPU.run() called before setSource()");

    if (IS_VIDEO_FRAME(this.source)) {
      this.runExternal(this.source);
    } else {
      this.run2d(this.source);
    }
  }

  private run2d(src: ImageBitmap): void {
    this.ensureStagingTexture(src.width, src.height);
    this.device.queue.copyExternalImageToTexture(
      { source: src, flipY: false },
      { texture: this.stagingTex! },
      [src.width, src.height],
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
    const enc  = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...this.dispatch);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
