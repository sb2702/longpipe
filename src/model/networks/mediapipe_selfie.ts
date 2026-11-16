import { runBatched, type Backend, type Tensor, type Op, type Activation } from '~/model/backend.ts'
import type { Conv2DWeights } from '~/model/weights.ts'

// Google's MediaPipe Selfie Segmenter (Apache-2.0 weights; see
// third_party/mediapipe/NOTICE.md), re-implemented on Longpipe's op library so it
// runs on the same WebGPU / WebGL2 backends, zero-copy Input op and compositor as
// Longpipe's own tiers — the network is the only difference.
//
// Graph: MobileNetV3-style encoder (hard-swish, squeeze-excite) → LR-ASPP head
// gated by global context → three decoder blocks with attention-gated skips →
// 2×2 stride-2 transposed conv → sigmoid. Mirrors the numpy reference in
// scripts/convert_mediapipe_selfie.py, which is checked against the TFLite
// interpreter (≤1e-5). Ported from the webrtcHacks selfie-segmentation demo
// (github.com/sb2702/webrtc-hacks-webgpu, segmentation/web/src/selfie/model.js).
//
// It is a SEGMENTATION model only: one alpha channel, no face / flow heads. So
// touch-up, auto-reframe and the optical-flow temporal stabilizer (which ride
// Longpipe's hydranet encoder) are not available with it.
//
// Output: 4-ch tensor at the input resolution, foreground probability in
// channel 0 (channels 1-3 are padding, sigmoid(0) = 0.5 — read .x only), i.e.
// the same contract as TierModel.output, so RenderOp can attach either.

export type MediaPipeSelfieVariant = 'square' | 'landscape'

export interface MediaPipeSelfieConfig {
  // Network input = alpha output resolution. The Input op stretches the frame
  // to this (MediaPipe does the same: no letterboxing).
  inputRes: { w: number; h: number }
  // Weight-pack filename, relative to the weights base URL.
  file:     string
}

export const MEDIAPIPE_SELFIE_CONFIG: Record<MediaPipeSelfieVariant, MediaPipeSelfieConfig> = {
  square:    { inputRes: { w: 256, h: 256 }, file: 'mediapipe_selfie_segmenter.f16.bin' },
  landscape: { inputRes: { w: 256, h: 144 }, file: 'mediapipe_selfie_segmenter_landscape.f16.bin' },
}

// Pick the variant whose input aspect distorts a w×h frame least (the Input op
// stretches, so the closer the aspect, the less the person is squashed):
// 16:9 → landscape; 1:1, portrait → square. 4:3 is exactly on the boundary
// (equal log-distortion either way) and goes to square, as do all ties.
export function pickMediaPipeSelfieVariant(w: number, h: number): MediaPipeSelfieVariant {
  const aspect = w / h
  const cost = (v: MediaPipeSelfieVariant) => {
    const r = MEDIAPIPE_SELFIE_CONFIG[v].inputRes
    return Math.abs(Math.log(aspect / (r.w / r.h)))
  }
  return cost('landscape') < cost('square') - 1e-9 ? 'landscape' : 'square'
}

// Weight pack shape (loadWeightsFromBinary of a mediapipe_selfie_*.bin): one
// { weights, bias } per TFLite layer, keyed by the original Keras layer name,
// already packed into the shader layout (conv: mat4x4[k²][O/4][I/4]; depthwise:
// vec4[k²][C/4]; biases padded to ×4). `meta` is informational.
export interface MediaPipeSelfieWeights {
  layers: Record<string, Conv2DWeights>
  meta?:  { model?: string; variant?: MediaPipeSelfieVariant; input?: { w: number; h: number }; [k: string]: unknown }
}

export class MediaPipeSelfie implements Op {
  readonly inputs: Tensor[]
  readonly output: Tensor
  private readonly steps: Op[] = []

  // input: (h, w, 4) RGB in [0, 1] (the Input op's output; alpha channel is
  // ignored — the first conv's 4th input channel has zero weights). h and w
  // must be multiples of 16; use MEDIAPIPE_SELFIE_CONFIG[variant].inputRes.
  constructor(private readonly backend: Backend, input: Tensor, private readonly w: MediaPipeSelfieWeights) {
    if (!w?.layers) throw new Error('MediaPipeSelfie: weights have no `layers` — expected a mediapipe_selfie_*.bin pack')
    if (input.h % 16 || input.w % 16)
      throw new Error(`MediaPipeSelfie: input ${input.w}×${input.h} must be a multiple of 16 (use MEDIAPIPE_SELFIE_CONFIG)`)
    const trained = w.meta?.input
    if (trained && (trained.w !== input.w || trained.h !== input.h))
      console.warn(`[longpipe] MediaPipeSelfie: input ${input.w}×${input.h} differs from the ${trained.w}×${trained.h} these weights were trained at`)
    this.inputs = [input]

    // Squeeze-excite: x * sigmoid(fc(relu(fc(mean(x)))))
    const squeezeExcite = (x: Tensor, fc1: string, fc2: string) =>
      this.scale(x, this.conv(this.conv(this.pool(x), fc1, 'relu'), fc2, 'sigmoid'))

    // MobileNetV3 block: 1×1 expand → depthwise → (SE) → 1×1 project (+ residual,
    // fused into the projection: ProjResidual)
    const invertedResidual = (x: Tensor, expand: string, dw: string, project: string,
      o: { stride?: number; se?: [string, string]; act?: Activation; residual?: boolean } = {}) => {
      const act = o.act ?? 'relu'
      let y = this.depthwise(this.conv(x, expand, act), dw, act, o.stride ?? 1)
      if (o.se) y = squeezeExcite(y, ...o.se)
      return o.residual ? this.projResidual(y, x, project) : this.conv(y, project)
    }

    // Upsample, blend with the encoder skip through an attention gate, refine
    // with a 1×1 conv and a residual depthwise conv. (Bilinear 2× + the 1×1
    // projection run fused: UpsampleConv1x1.)
    const decoderBlock = (x: Tensor, skip: Tensor, proj: string, [fc1, fc2]: [string, string], conv1: string, dw: string) => {
      const up = this.upsampleConv(x, proj)
      const gate = this.conv(this.conv(this.pool(this.add(skip, up)), fc1, 'relu'), fc2, 'sigmoid')
      const y = this.conv(this.add(this.scale(skip, gate), up), conv1, 'relu')
      return this.add(y, this.depthwise(y, dw, 'relu'))
    }

    // Encoder
    const s1 = this.conv(input, 'conv2d', 'hswish', 2)                                       // /2  ×16
    let x = this.depthwise(this.conv(s1, 'conv2d_1', 'relu'), 'depthwise_conv2d', 'relu', 2)
    x = squeezeExcite(x, 'conv2d_2', 'conv2d_3')
    const s2 = this.conv(x, 'conv2d_4')                                                       // /4  ×16
    x = invertedResidual(s2, 'conv2d_5', 'depthwise_conv2d_1', 'conv2d_6', { stride: 2 })    // /8  ×24
    const s3 = invertedResidual(x, 'conv2d_7', 'depthwise_conv2d_2', 'conv2d_8', { residual: true })
    x = invertedResidual(s3, 'conv2d_9', 'depthwise_conv2d_3', 'conv2d_12',                  // /16 ×32
      { stride: 2, se: ['conv2d_10', 'conv2d_11'], act: 'hswish' })
    for (let i = 0; i < 4; i++) {
      const n = 13 + 4 * i
      x = invertedResidual(x, `conv2d_${n}`, `depthwise_conv2d_${4 + i}`, `conv2d_${n + 3}`,
        { se: [`conv2d_${n + 1}`, `conv2d_${n + 2}`], act: 'hswish', residual: true })
    }

    // LR-ASPP head: features gated by a global-context branch               /16 ×128
    x = this.scale(this.conv(x, 'conv2d_29', 'relu'), this.conv(this.pool(x), 'conv2d_30', 'sigmoid'))

    // Decoder
    x = decoderBlock(x, s3, 'conv2d_31', ['conv2d_32', 'conv2d_33'], 'conv2d_34', 'depthwise_conv2d_8')   // /8
    x = decoderBlock(x, s2, 'conv2d_35', ['conv2d_36', 'conv2d_37'], 'conv2d_38', 'depthwise_conv2d_9')   // /4
    x = decoderBlock(x, s1, 'conv2d_39', ['conv2d_40', 'conv2d_41'], 'conv2d_42', 'depthwise_conv2d_10')  // /2

    // 2×2 stride-2 transposed conv → full res, sigmoid fused
    const seg = this.layer('segment')
    const up = backend.ops.ConvTranspose2d(x, seg, { outChannels: 4, kernel: 2, stride: 2, padding: 0, activation: 'sigmoid' })
    this.check('segment', seg.weights.length, 4 * 1 * (x.c / 4) * 16)
    this.steps.push(up)
    this.output = up.output
  }

  // One command buffer per run (see Backend.batch).
  run(): void {
    runBatched(this.backend, () => { for (const op of this.steps) op.run() })
  }

  // Number of GPU passes per run (WebGL's pool is two draws; counted as one op).
  get opCount(): number {
    return this.steps.length
  }

  // ── graph helpers: build an op, record it, return its output ────────────

  private layer(name: string): Conv2DWeights {
    const l = this.w.layers[name]
    if (!l) throw new Error(`MediaPipeSelfie: weights missing layer '${name}'`)
    return l
  }

  private check(name: string, got: number, want: number): void {
    if (got !== want) throw new Error(`MediaPipeSelfie: layer '${name}' has ${got} weights, expected ${want} — wrong weight file?`)
  }

  private push(op: Op): Tensor {
    this.steps.push(op)
    return op.output
  }

  private conv(x: Tensor, name: string, activation: Activation = 'none', stride = 1): Tensor {
    const l = this.layer(name)
    const outChannels = l.bias.length                       // padded to ×4 by the converter
    const kk = l.weights.length / ((outChannels / 4) * (x.c / 4) * 16)
    const kernel = Math.round(Math.sqrt(kk))
    this.check(name, l.weights.length, kernel * kernel * (outChannels / 4) * (x.c / 4) * 16)
    return this.push(this.backend.ops.Conv2d(x, l, { outChannels, kernel, stride, padding: 'same', activation }))
  }

  private depthwise(x: Tensor, name: string, activation: Activation, stride = 1): Tensor {
    const l = this.layer(name)
    const kernel = Math.round(Math.sqrt(l.weights.length / x.c))
    this.check(name, l.weights.length, kernel * kernel * x.c)
    this.check(name, l.bias.length, x.c)
    return this.push(this.backend.ops.DepthwiseConv2d(x, l, { kernel, stride, padding: 'same', activation }))
  }

  // 1×1 projection + residual add in one pass.
  private projResidual(x: Tensor, skip: Tensor, name: string): Tensor {
    const l = this.layer(name)
    const outChannels = l.bias.length
    this.check(name, l.weights.length, (outChannels / 4) * (x.c / 4) * 16)
    if (outChannels !== skip.c) throw new Error(`MediaPipeSelfie: '${name}' residual has ${skip.c} channels, expected ${outChannels}`)
    return this.push(this.backend.ops.ProjResidual(x, skip, l, { outChannels }))
  }

  // Bilinear 2× (half-pixel centres) + 1×1 conv, fused.
  private upsampleConv(x: Tensor, name: string): Tensor {
    const l = this.layer(name)
    const outChannels = l.bias.length
    this.check(name, l.weights.length, (outChannels / 4) * (x.c / 4) * 16)
    return this.push(this.backend.ops.UpsampleConv1x1(x, l, { outH: x.h * 2, outW: x.w * 2, outChannels, activation: 'none' }))
  }

  private pool(x: Tensor): Tensor       { return this.push(this.backend.ops.GlobalAvgPool(x)) }
  private scale(x: Tensor, s: Tensor)    { return this.push(this.backend.ops.ChannelScale(x, s)) }
  private add(a: Tensor, b: Tensor)      { return this.push(this.backend.ops.Add(a, b)) }
}
