import type { Tensor, Op, Activation } from '~/model/backend'

export class WebGPUTensor implements Tensor {
  constructor(
    readonly h: number,
    readonly w: number,
    readonly c: number,
    readonly buffer: GPUBuffer,
  ) {}
}

export abstract class WebGPUOp implements Op {
  abstract readonly inputs: Tensor[]
  abstract readonly output: WebGPUTensor
  protected abstract pipeline: GPUComputePipeline
  protected abstract bindGroup: GPUBindGroup
  protected abstract dispatchX: number
  protected abstract dispatchY: number
  protected abstract dispatchZ: number

  protected readonly device: GPUDevice

  constructor(device: GPUDevice) {
    this.device = device
  }

  run(): void {
    const enc = this.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.dispatchWorkgroups(this.dispatchX, this.dispatchY, this.dispatchZ)
    pass.end()
    this.device.queue.submit([enc.finish()])
  }
}

export function convOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: 'same' | 'valid',
): number {
  if (padding === 'same') return Math.ceil(inSize / stride)
  return Math.floor((inSize - kernel) / stride) + 1
}

// TF-style asymmetric SAME padding: returns the top/left pad (floor of total/2).
export function samePadHalf(inSize: number, outSize: number, kernel: number, stride: number): number {
  return Math.floor(Math.max((outSize - 1) * stride + kernel - inSize, 0) / 2)
}

export function activationCode(a: Activation): number {
  return a === 'relu6' ? 1 : 0
}

export function makeUniform(device: GPUDevice, values: number[]): GPUBuffer {
  const data = new Uint32Array(values)
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  new Uint32Array(buf.getMappedRange()).set(data)
  buf.unmap()
  return buf
}

// Pre-allocate a NHWC vec4 output buffer for an activation tensor.
export function makeOutputTensor(device: GPUDevice, h: number, w: number, c: number): WebGPUTensor {
  const buf = device.createBuffer({
    size: h * w * (c / 4) * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  return new WebGPUTensor(h, w, c, buf)
}

export function cast(t: Tensor): WebGPUTensor {
  return t as WebGPUTensor
}
