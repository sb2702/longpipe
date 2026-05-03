import type { Tensor, Conv2dParams } from "~/model/backend";
import type { WebGPUBackend } from "~/model/backends/webgpu/index";
import {
  WebGPUTensor,
  WebGPUOp,
  cast,
} from "~/model/backends/webgpu/base_webgpu_op";
import conv2dSrc from "~/model/backends/webgpu/shaders/conv2d.wgsl";

function convOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: number | "same" | "valid",
): number {
  if (typeof padding === "number") return Math.floor((inSize + 2 * padding - kernel) / stride) + 1;
  if (padding === "same") return Math.ceil(inSize / stride);
  return Math.floor((inSize - kernel) / stride) + 1;
}

function samePadHalf(
  inSize: number,
  outSize: number,
  kernel: number,
  stride: number,
): number {
  return Math.floor(Math.max((outSize - 1) * stride + kernel - inSize, 0) / 2);
}

export class Conv2DWebGPU extends WebGPUOp {
  readonly inputs: Tensor[];
  readonly output: WebGPUTensor;
  protected pipeline: GPUComputePipeline;
  protected bindGroup: GPUBindGroup;
  protected dispatchX: number;
  protected dispatchY: number;
  protected dispatchZ: number;

  constructor(
    backend: WebGPUBackend,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    params: Conv2dParams,
  ) {
    super(backend);
    const outH = convOutSize(
      input.h,
      params.kernel,
      params.stride,
      params.padding,
    );
    const outW = convOutSize(
      input.w,
      params.kernel,
      params.stride,
      params.padding,
    );
    const inGroups = input.c / 4;
    const outGroups = params.outChannels / 4;
    const padTop =
      typeof params.padding === "number"
        ? params.padding
        : params.padding === "same"
          ? samePadHalf(input.h, outH, params.kernel, params.stride)
          : 0;
    const padLeft =
      typeof params.padding === "number"
        ? params.padding
        : params.padding === "same"
          ? samePadHalf(input.w, outW, params.kernel, params.stride)
          : 0;

    this.output = backend.makeOutputTensor(outH, outW, params.outChannels);

    const uniformBuf = backend.makeUniform([
      input.h,
      input.w,
      outH,
      outW,
      inGroups,
      outGroups,
      params.kernel,
      params.kernel,
      params.stride,
      padTop,
      padLeft,
      params.activation === "relu6" ? 1 : 0,
    ]);

    this.pipeline = backend.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: backend.device.createShaderModule({ code: conv2dSrc }),
        entryPoint: "main",
      },
    });

    this.bindGroup = backend.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cast(input).buffer } },
        { binding: 1, resource: { buffer: cast(weights).buffer } },
        { binding: 2, resource: { buffer: cast(bias).buffer } },
        { binding: 3, resource: { buffer: this.output.buffer } },
        { binding: 4, resource: { buffer: uniformBuf } },
      ],
    });

    this.dispatchX = Math.ceil(outW / 8);
    this.dispatchY = Math.ceil(outH / 8);
    this.dispatchZ = outGroups;
    this.inputs = [input];
  }
}
