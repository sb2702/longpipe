import type { Backend, Tensor, Op } from '~/model/backend.ts'

// Pyramid blur — cascading 2× bilinear downsamples followed by 2× bilinear
// upsamples. Each bilinear pass is a 2×2 box filter; stacking N of them and
// going back up produces a smooth, gaussian-equivalent blur (central limit
// theorem). No dedicated convolution shader needed — reuses the
// BilinearUpsample op the backends already expose.
//
// Output stops one level short of input resolution (half of input dims). The
// consumer is expected to bilinearly sample the result during its own pass —
// see CompositeImageBilinear. This drops the final full-res upsample (the
// most expensive pyramid op by far) since the consumer's per-pixel scan
// absorbs that work for free.
//
// `sigma` keeps the same meaning as a Gaussian sigma in input pixels — we
// pick the pyramid depth so the lowest mip's pixel size approximates the
// desired blur radius (3σ).
export class BlurPyramid {
  private readonly downs: Op[] = []
  private readonly ups:   Op[] = []

  constructor(backend: Backend, input: Tensor, sigma: number) {
    const levels = pyramidLevels(sigma)

    let cur: Tensor = input
    for (let i = 0; i < levels; i++) {
      const op = backend.ops.BilinearUpsample(cur, {
        outH: Math.max(1, Math.floor(cur.h / 2)),
        outW: Math.max(1, Math.floor(cur.w / 2)),
      })
      this.downs.push(op)
      cur = op.output
    }

    // Up-chain stops at downs[0].output (half input resolution). The final
    // 2× upsample to full input res is intentionally omitted.
    for (let i = levels - 1; i >= 1; i--) {
      const target = this.downs[i - 1].output
      const op = backend.ops.BilinearUpsample(cur, { outH: target.h, outW: target.w })
      this.ups.push(op)
      cur = op.output
    }
  }

  get output(): Tensor {
    return this.ups.length > 0 ? this.ups[this.ups.length - 1].output : this.downs[0].output
  }

  run(): void {
    for (const op of this.downs) op.run()
    for (const op of this.ups)   op.run()
  }
}

// Effective blur radius after N pyramid levels ≈ 2^N input pixels. Solve for
// N from R ≈ 3σ. Clamped to keep extreme inputs sane (sigma < 1 collapses to
// 2 levels; sigma > ~20 caps at 6 levels = 64× total downsample).
function pyramidLevels(sigma: number): number {
  const targetRadius = Math.max(1, 3 * sigma)
  return Math.max(2, Math.min(6, Math.round(Math.log2(targetRadius))))
}
