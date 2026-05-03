import type { Backend } from "~/model/backend";
import type { WebGPUTensor, WebGPUMLBuffer } from "~/model/backends/webgpu/base_webgpu_op";
import { Conv2DWebGPU } from "~/model/backends/webgpu/ops/conv2d";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class WebGPUBackend implements Backend {
  readonly ops: Backend["ops"];

  private constructor(readonly device: GPUDevice) {
    this.ops = {
      Conv2d: (input, weights, bias, params) =>
        new Conv2DWebGPU(this, input, weights, bias, params),
    };
  }

  static async isAvailable(): Promise<boolean> {
    if (!navigator.gpu) return false;
    return (await navigator.gpu.requestAdapter()) !== null;
  }

  static async create(): Promise<WebGPUBackend> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU adapter not available");
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device);
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
