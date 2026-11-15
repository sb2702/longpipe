import type { Tensor, Op } from '~/model/backend.ts'
import type { WebGLBackend } from '~/model/backends/webgl/index.ts'
import { WebGLTensor, WebGLOp } from '~/model/backends/webgl/base_webgl_op.ts'
import poolRowsSrc from '~/model/backends/webgl/shaders/pool_rows.glsl'
import poolColsSrc from '~/model/backends/webgl/shaders/pool_cols.glsl'

class PoolRowsWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = poolRowsSrc

  constructor(backend: WebGLBackend, input: Tensor) {
    super(backend)
    const g = input.c / 4
    this.output = { h: input.h, w: 1, c: input.c, texture: this.makeTexture(null, g, input.h), texW: g, texH: input.h }
    this.inputs = [input]
    this.samplers = [{ name: 'u_input', texture: (input as WebGLTensor).texture }]
    this.uniformInts = { u_in_w: input.w, u_c_groups: g }
    this.defaultSetup()
    this.dispatch = [g, input.h]
  }
}

class PoolColsWebGL extends WebGLOp {
  readonly inputs: Tensor[]
  readonly weights: never[] = []
  readonly output: WebGLTensor
  protected dispatch: [number, number]
  shader = poolColsSrc

  constructor(backend: WebGLBackend, rows: WebGLTensor) {
    super(backend)
    const g = rows.c / 4
    this.output = { h: 1, w: 1, c: rows.c, texture: this.makeTexture(null, g, 1), texW: g, texH: 1 }
    this.inputs = [rows]
    this.samplers = [{ name: 'u_input', texture: rows.texture }]
    this.uniformInts = { u_in_h: rows.h }
    this.defaultSetup()
    this.dispatch = [g, 1]
  }
}

// (H, W, C) → (1, 1, C) mean, as two draws (row means, then their mean) —
// WebGL has no workgroup shared memory to reduce in one pass.
export class GlobalAvgPoolWebGL implements Op {
  readonly inputs: Tensor[]
  readonly output: WebGLTensor
  private readonly rows: PoolRowsWebGL
  private readonly cols: PoolColsWebGL

  constructor(backend: WebGLBackend, input: Tensor) {
    this.inputs = [input]
    this.rows = new PoolRowsWebGL(backend, input)
    this.cols = new PoolColsWebGL(backend, this.rows.output)
    this.output = this.cols.output
  }

  run(): void {
    this.rows.run()
    this.cols.run()
  }
}
