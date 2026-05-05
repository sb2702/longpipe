import type { Backend, Dtype, DataView_ } from "~/model/backend";
import type { WebGPUTensor, WebGPUMLBuffer } from "~/model/backends/webgpu/base_webgpu_op";
import { float32ArrayToHalf, halfArrayToFloat32 } from "~/utils/fp16";
import { Conv2DWebGPU } from "~/model/backends/webgpu/ops/conv2d";
import { DepthwiseConv2DWebGPU } from "~/model/backends/webgpu/ops/depthwise_conv2d";
import { AddWebGPU } from "~/model/backends/webgpu/ops/add";
import { SigmoidWebGPU } from "~/model/backends/webgpu/ops/sigmoid";
import { BilinearUpsampleWebGPU } from "~/model/backends/webgpu/ops/bilinear_upsample";
import { BicubicUpsampleWebGPU  } from "~/model/backends/webgpu/ops/bicubic_upsample";
import { ChannelConcatWebGPU } from "~/model/backends/webgpu/ops/channel_concat";
import { Conv2dAddWebGPU } from "~/model/backends/webgpu/ops/conv2d_add";
import { UpsampleConcatWebGPU } from "~/model/backends/webgpu/ops/upsample_concat";
import { UpsampleConv1x1WebGPU } from "~/model/backends/webgpu/ops/upsample_conv1x1";
import { UpsampleSigmoidWebGPU } from "~/model/backends/webgpu/ops/upsample_sigmoid";
import { CompositeSolidWebGPU } from "~/model/backends/webgpu/ops/composite_solid";
import { CompositeImageWebGPU } from "~/model/backends/webgpu/ops/composite_image";
import { CompositeImageBilinearWebGPU } from "~/model/backends/webgpu/ops/composite_image_bilinear";
import { CompositePassthroughWebGPU } from "~/model/backends/webgpu/ops/composite_passthrough";
import { InputWebGPU } from "~/model/backends/webgpu/ops/input";

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
  readonly canvasContext: GPUCanvasContext;
  readonly canvasFormat: GPUTextureFormat;
  readonly bytesPerElement: 2 | 4;

  private constructor(
    readonly device: GPUDevice,
    readonly canvas: HTMLCanvasElement | OffscreenCanvas,
    readonly dtype: Dtype,
  ) {
    this.bytesPerElement = dtype === "f16" ? 2 : 4;

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
      BicubicUpsample:  (input, params)                 => new BicubicUpsampleWebGPU(this, input, params),
      ChannelConcat:    (a, b)                          => new ChannelConcatWebGPU(this, a, b),
      Conv2dAdd:        (input, skip, weights, params)  => new Conv2dAddWebGPU(this, input, skip, weights, params),
      UpsampleConcat:   (a, b, params)                  => new UpsampleConcatWebGPU(this, a, b, params),
      UpsampleConv1x1:  (input, weights, params)        => new UpsampleConv1x1WebGPU(this, input, weights, params),
      UpsampleSigmoid:  (input, params)                 => new UpsampleSigmoidWebGPU(this, input, params),
      Input:            (h, w)                          => new InputWebGPU(this, h, w),
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
      CompositeImage: (image, alpha, bg) => {
        const op = new CompositeImageWebGPU(this, image, alpha, bg);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture());
            op.run();
          },
        };
      },
      CompositeImageBilinear: (image, alpha, bg) => {
        const op = new CompositeImageBilinearWebGPU(this, image, alpha, bg);
        return {
          run: () => {
            op.setOutput(this.getCurrentDisplayTexture());
            op.run();
          },
        };
      },
      CompositePassthrough: (image) => {
        const op = new CompositePassthroughWebGPU(this, image);
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

  // The swapchain texture for the current frame. Must be called inside the
  // same task that submits the render commands — texture is invalidated after
  // the next browser paint.
  getCurrentDisplayTexture(): GPUTexture {
    return this.canvasContext.getCurrentTexture();
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

  // Awaits all in-flight queue work. Used for benchmarking when we want
  // a sync barrier without paying the readback bandwidth cost.
  async sync(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  destroy(): void {
    this.device.destroy();
  }
}
