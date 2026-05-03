import type { Backend } from '~/model/backend'
import { WebGPUTensor, makeOutputTensor } from '~/model/backends/webgpu/base'
import { Conv2DWebGPU } from '~/model/backends/webgpu/ops/conv2d'
import { DepthwiseConv2DWebGPU } from '~/model/backends/webgpu/ops/depthwise_conv2d'
import { AddWebGPU } from '~/model/backends/webgpu/ops/add'
import { UpsampleWebGPU } from '~/model/backends/webgpu/ops/upsample'
import { ConcatWebGPU } from '~/model/backends/webgpu/ops/concat'
import { SigmoidWebGPU } from '~/model/backends/webgpu/ops/sigmoid'
import { Conv2dAddWebGPU } from '~/model/backends/webgpu/ops/conv2d_add'
import { UpsampleConcatWebGPU } from '~/model/backends/webgpu/ops/upsample_concat'
import { UpsampleSigmoidWebGPU } from '~/model/backends/webgpu/ops/upsample_sigmoid'
import { UpsampleConv1x1WebGPU } from '~/model/backends/webgpu/ops/upsample_conv1x1'

import conv2dSrc          from '~/model/backends/webgpu/shaders/conv2d.wgsl'
import depthwiseSrc       from '~/model/backends/webgpu/shaders/depthwise_conv2d.wgsl'
import addSrc             from '~/model/backends/webgpu/shaders/add.wgsl'
import upsampleSrc        from '~/model/backends/webgpu/shaders/bilinear_upsample.wgsl'
import concatSrc          from '~/model/backends/webgpu/shaders/channel_concat.wgsl'
import sigmoidSrc         from '~/model/backends/webgpu/shaders/sigmoid.wgsl'
import conv2dAddSrc       from '~/model/backends/webgpu/shaders/conv2d_add.wgsl'
import upsampleConcatSrc  from '~/model/backends/webgpu/shaders/upsample_concat.wgsl'
import upsampleSigmoidSrc from '~/model/backends/webgpu/shaders/upsample_sigmoid.wgsl'
import upsampleConv1x1Src from '~/model/backends/webgpu/shaders/upsample_conv1x1.wgsl'

type Pipelines = {
  conv2d:          GPUComputePipeline
  depthwiseConv2d: GPUComputePipeline
  add:             GPUComputePipeline
  upsample:        GPUComputePipeline
  concat:          GPUComputePipeline
  sigmoid:         GPUComputePipeline
  conv2dAdd:       GPUComputePipeline
  upsampleConcat:  GPUComputePipeline
  upsampleSigmoid: GPUComputePipeline
  upsampleConv1x1: GPUComputePipeline
}

export class WebGPUBackend implements Backend {
  readonly ops: Backend['ops']

  private constructor(
    readonly device: GPUDevice,
    readonly pipelines: Pipelines,
  ) {
    const d = device
    const p = pipelines
    this.ops = {
      Conv2d:          (input, weights, bias, params) =>
        new Conv2DWebGPU(d, p.conv2d, input, weights, bias, params),
      DepthwiseConv2d: (input, weights, bias, params) =>
        new DepthwiseConv2DWebGPU(d, p.depthwiseConv2d, input, weights, bias, params),
      Add:             (inputs) =>
        new AddWebGPU(d, p.add, inputs),
      Upsample:        (input, outH, outW) =>
        new UpsampleWebGPU(d, p.upsample, input, outH, outW),
      Concat:          (inputs) =>
        new ConcatWebGPU(d, p.concat, inputs),
      Sigmoid:         (input) =>
        new SigmoidWebGPU(d, p.sigmoid, input),
      Conv2dAdd:       (input, residual, weights, bias, params) =>
        new Conv2dAddWebGPU(d, p.conv2dAdd, input, residual, weights, bias, params),
      UpsampleConcat:  (input, skip, outH, outW) =>
        new UpsampleConcatWebGPU(d, p.upsampleConcat, input, skip, outH, outW),
      UpsampleSigmoid: (input, outH, outW) =>
        new UpsampleSigmoidWebGPU(d, p.upsampleSigmoid, input, outH, outW),
      UpsampleConv1x1: (input, weights, bias, activation, outH, outW, outChannels) =>
        new UpsampleConv1x1WebGPU(d, p.upsampleConv1x1, input, weights, bias, activation, outH, outW, outChannels),
    }
  }

  static async isAvailable(): Promise<boolean> {
    if (!navigator.gpu) return false
    const adapter = await navigator.gpu.requestAdapter()
    return adapter !== null
  }

  static async create(): Promise<WebGPUBackend> {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('WebGPU adapter not available')
    const device = await adapter.requestDevice()

    const makePipeline = (src: string): Promise<GPUComputePipeline> => {
      const module = device.createShaderModule({ code: src })
      return device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      })
    }

    const [
      conv2d, depthwiseConv2d, add, upsample, concat,
      sigmoid, conv2dAdd, upsampleConcat, upsampleSigmoid, upsampleConv1x1,
    ] = await Promise.all([
      makePipeline(conv2dSrc),
      makePipeline(depthwiseSrc),
      makePipeline(addSrc),
      makePipeline(upsampleSrc),
      makePipeline(concatSrc),
      makePipeline(sigmoidSrc),
      makePipeline(conv2dAddSrc),
      makePipeline(upsampleConcatSrc),
      makePipeline(upsampleSigmoidSrc),
      makePipeline(upsampleConv1x1Src),
    ])

    return new WebGPUBackend(device, {
      conv2d, depthwiseConv2d, add, upsample, concat,
      sigmoid, conv2dAdd, upsampleConcat, upsampleSigmoid, upsampleConv1x1,
    })
  }

  // Upload a Float32Array to GPU as a weight tensor (spatial dims not meaningful for weights).
  upload(data: Float32Array): WebGPUTensor {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    })
    new Float32Array(buf.getMappedRange()).set(data)
    buf.unmap()
    return new WebGPUTensor(0, 0, 0, buf)
  }

  // Create a writable input tensor (for writing the frame before each forward pass).
  createTensor(h: number, w: number, c: number): WebGPUTensor {
    return makeOutputTensor(this.device, h, w, c)
  }

  // Read tensor data back to CPU (used in tests).
  async readback(tensor: WebGPUTensor): Promise<Float32Array> {
    const size = tensor.buffer.size
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const enc = this.device.createCommandEncoder()
    enc.copyBufferToBuffer(tensor.buffer, 0, staging, 0, size)
    this.device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const result = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return result
  }

  destroy(): void {
    this.device.destroy()
  }
}
