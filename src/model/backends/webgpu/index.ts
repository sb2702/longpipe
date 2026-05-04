import type { Backend } from "~/model/backend";
import type { WebGPUTensor, WebGPUMLBuffer } from "~/model/backends/webgpu/base_webgpu_op";
import { Conv2DWebGPU } from "~/model/backends/webgpu/ops/conv2d";
import { DepthwiseConv2DWebGPU } from "~/model/backends/webgpu/ops/depthwise_conv2d";
import { AddWebGPU } from "~/model/backends/webgpu/ops/add";
import { SigmoidWebGPU } from "~/model/backends/webgpu/ops/sigmoid";
import { BilinearUpsampleWebGPU } from "~/model/backends/webgpu/ops/bilinear_upsample";
import { ChannelConcatWebGPU } from "~/model/backends/webgpu/ops/channel_concat";
import { Conv2dAddWebGPU } from "~/model/backends/webgpu/ops/conv2d_add";
import { UpsampleConcatWebGPU } from "~/model/backends/webgpu/ops/upsample_concat";
import { UpsampleConv1x1WebGPU } from "~/model/backends/webgpu/ops/upsample_conv1x1";
import { UpsampleSigmoidWebGPU } from "~/model/backends/webgpu/ops/upsample_sigmoid";
import { CompositeSolidWebGPU } from "~/model/backends/webgpu/ops/composite_solid";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export interface WebGPUBackendOptions {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  device?: GPUDevice;
}

export class WebGPUBackend implements Backend {
  readonly ops: Backend["ops"];
  readonly presenters: Backend["presenters"];
  readonly canvasContext: GPUCanvasContext;
  readonly canvasFormat: GPUTextureFormat;

  private constructor(
    readonly device: GPUDevice,
    readonly canvas: HTMLCanvasElement | OffscreenCanvas,
  ) {
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Error("Failed to get WebGPU context from canvas");
    this.canvasContext = ctx;
    this.canvasFormat  = navigator.gpu.getPreferredCanvasFormat();
    this.canvasContext.configure({
      device,
      format: this.canvasFormat,
      alphaMode: "premultiplied",
    });

    this.ops = {
      Conv2d:           (input, weights, params)        => new Conv2DWebGPU(this, input, weights, params),
      DepthwiseConv2d:  (input, weights, params)        => new DepthwiseConv2DWebGPU(this, input, weights, params),
      Add:              (a, b)                          => new AddWebGPU(this, a, b),
      Sigmoid:          (input)                         => new SigmoidWebGPU(this, input),
      BilinearUpsample: (input, params)                 => new BilinearUpsampleWebGPU(this, input, params),
      ChannelConcat:    (a, b)                          => new ChannelConcatWebGPU(this, a, b),
      Conv2dAdd:        (input, skip, weights, params)  => new Conv2dAddWebGPU(this, input, skip, weights, params),
      UpsampleConcat:   (a, b, params)                  => new UpsampleConcatWebGPU(this, a, b, params),
      UpsampleConv1x1:  (input, weights, params)        => new UpsampleConv1x1WebGPU(this, input, weights, params),
      UpsampleSigmoid:  (input, params)                 => new UpsampleSigmoidWebGPU(this, input, params),
    };

    this.presenters = {
      // The op needs the per-frame swapchain texture before each draw — wrap
      // it so callers don't have to think about presentation lifecycle.
      CompositeSolid: (image, alpha, bgColor) => {
        const op = new CompositeSolidWebGPU(this, image, alpha, bgColor);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture());
            op.run();
          },
        };
      },
    };
  }

  static async isAvailable(): Promise<boolean> {
    if (!navigator.gpu) return false;
    return (await navigator.gpu.requestAdapter()) !== null;
  }

  static async create(opts: WebGPUBackendOptions): Promise<WebGPUBackend> {
    let device = opts.device;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("WebGPU adapter not available");
      device = await adapter.requestDevice();
    }
    return new WebGPUBackend(device, opts.canvas);
  }

  // The swapchain texture for the current frame. Must be called inside the
  // same task that submits the render commands — texture is invalidated after
  // the next browser paint.
  getCurrentDisplayTexture(): GPUTexture {
    return this.canvasContext.getCurrentTexture();
  }

  tensor(h: number, w: number, c: number, data?: Float32Array): WebGPUTensor {
    const size = h * w * c * 4;
    const buf = this.device.createBuffer({
      size,
      usage: STORAGE,
      mappedAtCreation: data !== undefined,
    });
    if (data) {
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
    }
    return { h, w, c, buffer: buf };
  }

  upload(data: Float32Array): WebGPUMLBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return { buffer: buf };
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
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  destroy(): void {
    this.device.destroy();
  }
}
