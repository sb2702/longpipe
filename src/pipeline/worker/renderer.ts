// Transport-agnostic core. Wraps RenderOp from sdk/src/model/render_op.ts
// with frame-skipping logic. Adapters call process(frame) per input frame;
// renderer decides whether to run the model this frame.

import type { Backend, Tensor, InputOp, Op, RenderTarget } from '~/model/backend.ts'
// Note: this file builds the GPU compute chain (network + RenderOp). The
// Streams API pipe chain (inputReadable → transform → outputWritable) is
// wired in worker/index.ts handleInit() where transport adapters are
// composed with this renderer.
import { RenderOp, type BackgroundConfig as RenderOpBackgroundConfig, type CompositeSpec } from '~/model/render_op.ts'
import { loadWeightsFromBinary } from '~/utils/loadWeights.ts'
import { TierModel } from '~/model/tier_model.ts'
import { OpticalFlowNet } from '~/model/networks/optical_flow_net.ts'
import { TIER_CONFIG } from '~/model/tier_config.ts'

const FLOW_DEC_W = 16   // shipping flow-head width
// Flow-gated stabilizer params. tLo/tHi are flow magnitudes in BASE-res pixels
// (the flow predicts base-res-unit displacements); envelope peak-hold mirrors the
// eval harness's known-good (leak 0.2, release 0.9). Tunable.
// tDiv/divScale: the occlusion-seam term — the gate also opens where |div(flow)|
// exceeds tDiv (a boundary the magnitude gate is blind to, since the revealed
// background is static). stepX/stepY are added per-tier in setPreset (≈ canvas/flow
// resolution ratio so the finite-difference spans one base/4 pixel). Tunable.
const FLOW_STAB = { tLo: 0.15, tHi: 2.5, leak: 0.15, release: 0.9, tDiv:1.0, divScale: 2.0 }
import type { ManualPreset } from '../presets'
import type { Background } from '../background'
import type { RendererStats } from '../messages'
import type { Topology } from '../topology'

// Preview: render a *candidate* effect to a second canvas while the main
// outgoing stream keeps its currently-applied effect. The network + alpha are
// shared (computed once); only the compositor differs. Lower-priority than
// main — throttled to previewFps and only composited on preview ticks.
export interface PreviewOptions {
  fps?: number   // default DEFAULT_PREVIEW_FPS — preview composite cadence
}

const DEFAULT_PREVIEW_FPS = 15

export interface RendererOptions {
  backend:     Backend
  backendKind: 'webgpu' | 'webgl'    // gates GPU-time sampling (WebGL's fence polling is too heavy for per-frame use)
  canvas:      OffscreenCanvas
  background:  Background
  enabled:     boolean
  topology:    Topology              // reported in stats for debug
}

// Renderer is constructed in boot mode (no preset / no weights / no network).
// Pipeline starts processing frames immediately in passthrough — that lets
// the consumer see live video during autotune + weight fetch (1-3s on
// modest hardware). Call setPreset(preset, weights) once the choice is
// resolved + weights are loaded; the renderer attaches the network in
// place and switches to effect mode mid-pipe (no pipe restart needed).

// Only sample real GPU-time every Nth model run, and only on WebGPU.
// WebGPU sync is one native promise per call (cheap); WebGL sync is a
// setTimeout(1ms) polling loop per call (multiple in flight = many
// concurrent loops). Sampling 1-in-5 keeps the adaptive logic fed without
// adding measurement overhead that would itself perturb the measurement.
const MODEL_TIMING_SAMPLE_INTERVAL = 5

const STATS_WINDOW_MS = 1000

export class Renderer {
  readonly canvas: OffscreenCanvas

  private readonly backend:     Backend
  private readonly backendKind: 'webgpu' | 'webgl'
  private readonly topology:    Topology
  // null until the first setPreset call. Stats use a 'boot' placeholder
  // while null; process() runs passthrough.
  private preset:  ManualPreset | null = null
  private enabled: boolean
  // Current background tracked here so setPreset can rebuild the RenderOp
  // with the same background (otherwise an adaptive swap would flash to
  // a hardcoded fallback).
  private currentBackground: Background
  private modelTimingCounter: number = 0

  private renderOp:     RenderOp
  // Optical-flow temporal state — null unless the loaded .bin carries a `flow` blob.
  // The renderer owns it (carriers + base-res inputs), like the old GRU hidden.
  // Flow is forward (prev→current): frame-a = last matting frame, frame-b = current.
  // Every frame the fresh alpha (matting on inference / warp on skips) is gated
  // against the warp-aligned previous stabilized alpha.
  //   everyFrame : skipFrames===0 (large/xl) → run flow BEFORE matting each frame
  //   warm       : false until the first inference seeds the carrier
  //   curBaseDown: current frame at base res = canvas frame downsampled (flow
  //                frame-b). NOT a direct source→base resample — that aliases.
  //   frameAHeld : last matting frame at base res (flow frame-a)
  //   net/up     : flow → base→canvas spatial upsample (values stay base-res-px)
  //   predBuf    : fresh per-frame alpha (matting or warp), canvas res
  //   stabPrev   : previous stabilized alpha (.x) + envelope (.y), canvas res
  //   refWarp    : warp_prev(stabPrev) — the gate's held reference, every frame
  //   stab       : flow-gated blend g·pred + (1-g)·ref
  //   alphaHeld/predWarp : skip-tier only — last inference alpha + its skip-frame warp
  private flow: {
    tier: TierModel; everyFrame: boolean; warm: boolean
    networkInput: InputOp; curBaseDown: Op; frameAHeld: Tensor
    net: OpticalFlowNet; up: Op
    predBuf: Tensor; stabPrev: Tensor; refWarp: Op; stab: Op
    alphaHeld: Tensor | null; predWarp: Op | null
  } | null = null
  // Per-target Input ops + ports for 'image'/'video' background modes. Keyed by
  // RenderTarget so main and preview can each show a different image/video bg
  // without colliding on one input op. Each video port receives a fresh
  // VideoFrame from main per frame; we upload it into that target's input
  // tensor and close the frame. The compositor reads input.output every render
  // (last write wins — no sync needed).
  private bgImageInputs = new Map<RenderTarget, InputOp>()
  private bgVideoInputs = new Map<RenderTarget, InputOp>()
  private bgVideoPorts  = new Map<RenderTarget, MessagePort>()

  // Preview state. previewCanvas is set once by attachPreview(); previewBg is
  // the candidate effect (null = not previewing — clearPreview()). previewCtx
  // is the WebGL present target (bitmaprenderer); WebGPU composites straight to
  // the attached canvas instead. previewEffectSpec is the precomputed effect
  // config (null when the candidate is 'none' → passthrough preview).
  private previewCanvas:     OffscreenCanvas | null = null
  private previewCtx:        ImageBitmapRenderingContext | null = null
  private previewBg:         Background | null = null
  private previewEffectSpec: RenderOpBackgroundConfig | null = null
  private previewIntervalMs: number = 1000 / DEFAULT_PREVIEW_FPS
  private lastPreviewAt:     number = 0

  // Frame-skipping: counter-based per preset.skipFrames. Model runs when
  // counter hits 0; counter is reset to skipFrames after each run and
  // decremented every other frame. Counter starts at 0 so first frame
  // always runs the model (warm alpha tensor for compositor).
  private skipCounter: number = 0

  // Stats — rolling windows trimmed to STATS_WINDOW_MS. Duration arrays
  // are stored as { ts, ms } pairs so the trim-by-time logic can compare
  // against a timestamp; previous shape (raw ms numbers) was getting
  // wiped every poll because trimRecent treated the durations as
  // timestamps and dropped them all.
  private framesRenderedAt: number[]                       = []
  private modelRunSamples:   { ts: number; ms: number }[]   = []
  private displayRunSamples: { ts: number; ms: number }[]   = []
  private skippedCount:      number                          = 0

  constructor(opts: RendererOptions) {
    this.backend           = opts.backend
    this.backendKind       = opts.backendKind
    this.canvas            = opts.canvas
    this.enabled           = opts.enabled
    this.currentBackground = opts.background
    this.topology          = opts.topology

    // Boot mode — RenderOp builds only displayInput + passthroughCompositor.
    // The network is attached later via setPreset() once weights arrive.
    this.renderOp = new RenderOp(this.backend)
  }

  // Per-frame entry point. Always cheap (display). Sometimes expensive (model).
  //
  // Handles the main × preview effect matrix: each surface independently runs
  // an effect composite (background.kind !== 'none') or a raw passthrough
  // (kind === 'none', or disabled / still booting). The model runs iff EITHER
  // surface needs alpha — so a preview effect forces inference even when the
  // main output is passthrough (matrix cell b).
  process(frame: VideoFrame): void {
    this.renderOp.setSource(frame)
    const now = performance.now()

    const hasNet        = this.renderOp.hasNetwork()
    const mainEffect    = this.enabled && hasNet && this.currentBackground.kind !== 'none'
    const previewTick   = this.isPreviewTick(now)
    const previewEffect = previewTick && hasNet && this.previewBg !== null && this.previewBg.kind !== 'none'


    // Model gate. Flow tiers run the matting + flow + stabilizer in stepFlow;
    // non-flow tiers follow the plain skip cadence. A preview tick that needs alpha
    // forces a run when main didn't already produce one this frame.
    let ranModel = false
    if (mainEffect && this.flow) {
      ranModel = this.stepFlow()
    } else if (mainEffect) {
      if (this.shouldRunModel()) { this.runModelOnce(); ranModel = true }
      else this.skippedCount++
    }
    if (!ranModel && previewEffect) { this.runModelOnce(); ranModel = true }

    // Composite. The display input (full-res image, shared by both surfaces) is
    // refreshed once inside whichever composite path runs first.
    const td = performance.now()
    if (previewTick && this.previewCanvas) {
      if (this.backendKind === 'webgl') {
        // WebGL: one canvas — preview composited + snapshotted, then main last
        // so the output adapter captures main. All synchronous (no race).
        this.compositePreviewWebGL(mainEffect, previewEffect)
      } else {
        // WebGPU: main + preview render to independent swapchains; order free.
        this.compositeMain(mainEffect)
        this.renderOp.compositeTo('preview', this.previewSpec(previewEffect))
      }
      this.lastPreviewAt = now
    } else {
      this.compositeMain(mainEffect)
    }
    this.displayRunSamples.push({ ts: performance.now(), ms: performance.now() - td })
    this.trimSamples(this.displayRunSamples)

    this.framesRenderedAt.push(performance.now())
    this.trim(this.framesRenderedAt)
  }

  // Main surface: effect (runDisplay) or passthrough. Both refresh the shared
  // display input as their first step.
  private compositeMain(effect: boolean): void {
    if (effect) this.renderOp.runDisplay()
    else        this.renderOp.runPassthrough()
  }

  // WebGL preview present: render preview to the single canvas, snapshot it to
  // the preview canvas's bitmaprenderer context, then re-composite main so the
  // canvas rests on main content. transferToImageBitmap detaches the canvas
  // backing (left blank), but main is recomposited before this synchronous
  // block ends, so neither the output adapter nor a continuous captureStream
  // ever observes preview/blank content.
  private compositePreviewWebGL(mainEffect: boolean, previewEffect: boolean): void {
    this.renderOp.refreshDisplayInput()
    this.renderOp.compositeTo('preview', this.previewSpec(previewEffect))
    const bmp = this.canvas.transferToImageBitmap()   // this.canvas === backend.canvas, typed OffscreenCanvas
    this.previewCtx!.transferFromImageBitmap(bmp)
    this.renderOp.compositeMain(!mainEffect)   // main last; passthrough when no main effect
  }

  // Preview compositor spec: the precomputed effect config, or passthrough when
  // the candidate is 'none' / inference isn't available.
  private previewSpec(previewEffect: boolean): CompositeSpec {
    return previewEffect && this.previewEffectSpec
      ? this.previewEffectSpec
      : { mode: 'passthrough' }
  }

  // Run the model once. All tiers are static — temporal stability comes from the
  // separate optical-flow path, not recurrent state threaded here.
  private runModelOnce(): void {
    const t = performance.now()
    this.renderOp.runModel()
    // GPU-time sampling: WebGPU only (cheap native promise per sync; WebGL's
    // setTimeout-polling fence is too heavy at per-frame rate). Even on WebGPU,
    // sample 1 in N runs to keep concurrent in-flight syncs bounded.
    if (this.backendKind === 'webgpu' && this.modelTimingCounter % MODEL_TIMING_SAMPLE_INTERVAL === 0) {
      this.backend.sync().then(() => {
        this.modelRunSamples.push({ ts: performance.now(), ms: performance.now() - t })
        this.trimSamples(this.modelRunSamples)
      }).catch(() => { /* sync errors during shutdown are expected */ })
    }
    this.modelTimingCounter++
  }

  // One flow-tier frame: forward flow (prev→current) → a fresh alpha (matting on
  // inference frames, warp on skips) → gate it against the warp-aligned previous
  // stabilized alpha. The gate reads the last-available flow:
  //   every-frame tiers (large/xl): run flow BEFORE matting (taps still hold prev)
  //   skip tiers: flow runs on skip frames; inference frames reuse the cached flow
  // Returns whether the matting model ran this frame.
  private stepFlow(): boolean {
    const f = this.flow!
    let ranModel = false

    // Resample the current frame to canvas, then downsample to base for the flow
    // stem (matches training). runModelOnce re-runs networkInput — cheap + harmless.
    f.networkInput.run()
    f.curBaseDown.run()

    if (f.everyFrame) {
      f.net.run(); f.up.run()                                  // before matting → taps are prev's
      this.runModelOnce(); ranModel = true
      this.backend.copyTensor(f.curBaseDown.output, f.frameAHeld)  // current = next frame-a
      this.backend.copyTensor(f.tier.output, f.predBuf)
    } else if (this.shouldRunModel()) {                        // skip tier, inference frame
      this.runModelOnce(); ranModel = true
      this.backend.copyTensor(f.curBaseDown.output, f.frameAHeld)
      this.backend.copyTensor(f.tier.output, f.alphaHeld!)     // warp source for the coming skips
      this.backend.copyTensor(f.tier.output, f.predBuf)
      // flow not recomputed — the gate reuses f.up's cached (last skip) flow
    } else {                                                   // skip tier, skip frame
      this.skippedCount++
      f.net.run(); f.up.run()                                  // flow = last-inference → current
      f.predWarp!.run()
      this.backend.copyTensor(f.predWarp!.output, f.predBuf)   // pred = warped last inference
    }

    if (!f.warm) {                                             // first inference seeds the carrier;
      this.backend.copyTensor(f.tier.output, f.stabPrev)       // matting is already in upscaler.output
      f.warm = ranModel                                        // (runModelOnce ran it) — no applyAlpha
      return ranModel
    }

    f.refWarp.run()                                            // ref = warp_prev(stabPrev)
    f.stab.run()                                               // alpha_stab = g·pred + (1-g)·ref
    this.backend.copyTensor(f.stab.output, f.stabPrev)         // threads stab (.x) + env (.y)
    this.renderOp.applyAlpha(f.stab.output)
    return ranModel
  }

  // True when the preview should composite this frame: a canvas is attached, a
  // candidate is set (null = clearPreview), the SDK is enabled, and enough time
  // has elapsed since the last preview composite (throttle — main is priority).
  private isPreviewTick(now: number): boolean {
    if (!this.previewCanvas || this.previewBg === null || !this.enabled) return false
    return now - this.lastPreviewAt >= this.previewIntervalMs
  }

  private shouldRunModel(): boolean {
    if (this.skipCounter === 0) {
      this.skipCounter = this.preset?.skipFrames ?? 0
      return true
    }
    this.skipCounter--
    return false
  }

  setBackground(bg: Background): void {
    this.currentBackground = bg
    // RenderOp.setBackground is a no-op when no network is attached
    // (config gets stored, applied on next setPreset).
    this.renderOp.setBackground(this.translateBackgroundFor('main', bg))
  }

  setEnabled(on: boolean): void {
    this.enabled = on
  }

  // Attach the preview output canvas (once — the canvas's control was
  // transferred from main). WebGPU registers a 2nd GPUCanvasContext on the
  // shared device and composites the preview straight to it; WebGL can't drive
  // a 2nd canvas, so we present via its bitmaprenderer context (a snapshot of
  // the main canvas — see compositePreviewWebGL).
  attachPreview(canvas: OffscreenCanvas): void {
    // The compositors don't resample (output index uses image width), so the
    // preview canvas backing MUST match the main canvas resolution. We own the
    // (transferred) offscreen, so resize it here rather than make the caller
    // guess the output resolution — the app CSS-scales the visible element.
    canvas.width  = this.canvas.width
    canvas.height = this.canvas.height
    this.previewCanvas = canvas
    if (this.backendKind === 'webgpu') {
      this.backend.attachCanvas('preview', canvas)
    } else {
      const ctx = canvas.getContext('bitmaprenderer')
      if (!ctx) throw new Error('renderer: failed to get bitmaprenderer context for preview canvas')
      this.previewCtx = ctx
    }
  }

  // Set / update the previewed candidate effect — same canonical Background as
  // the main effect. fps throttles the preview composite (main stays priority).
  // 'none' is a valid candidate: previews the no-effect (raw) option.
  setPreview(bg: Background, opts?: PreviewOptions): void {
    this.previewBg         = bg
    this.previewEffectSpec = bg.kind === 'none' ? null : this.translateBackgroundFor('preview', bg)
    if (opts?.fps && opts.fps > 0) this.previewIntervalMs = 1000 / opts.fps
    this.lastPreviewAt     = 0   // composite on the very next frame
  }

  // Stop previewing — the preview canvas freezes on its last frame (the app can
  // hide it). The model gate drops back to main-only. The attached canvas stays
  // attached for a future setPreview.
  clearPreview(): void {
    this.previewBg         = null
    this.previewEffectSpec = null
    const port = this.bgVideoPorts.get('preview')
    if (port) { port.close(); this.bgVideoPorts.delete('preview') }
  }

  // Attach the network — first call wires it from boot mode, subsequent
  // calls swap presets at runtime (used by adaptive controller). The
  // RenderOp instance is reused: only the network-dependent pieces
  // (network, networkInput, upscaler, compositor) are rebuilt.
  setPreset(preset: ManualPreset, weightsBuf: ArrayBuffer): void {
    const cfg = TIER_CONFIG[preset.model]
    if (!cfg) throw new Error(`renderer: no TIER_CONFIG entry for model '${preset.model}'`)

    // Composite tier weights { base, wrapper, gru? }. loadWeightsFromBinary
    // resolves the tree generically; tier_config (model-as-code) decides the
    // structure / placement, NOT the .bin. Cast: the resolved shape is the
    // union of base ModelWeights + wrapper + optional gru.
    const w = loadWeightsFromBinary(weightsBuf) as any

    // x_hr at canvas res — authoritative from tier_config, not preset.resolution
    // (the wrapper down-path strides this to base res internally).
    const networkInput = this.backend.ops.Input(cfg.canvasRes.h, cfg.canvasRes.w)

    const tier = new TierModel(
      this.backend, networkInput.output, w.base, w.wrapper, cfg.wrapper, cfg.base,
    )

    this.renderOp.attachNetwork(tier, networkInput, {
      upscaler:   'bilinear',
      background: this.translateBackgroundFor('main', this.currentBackground),
    })

    // Optical-flow temporal: wired iff the .bin carries a `flow` blob. The flow net
    // rides the tier's cached encoder taps; warps run at canvas res (network output).
    // The net predicts BACKWARD flow (frame-b→frame-a), so the warp gathers at
    // p + flow: flowScale = +(canvasW/baseW) (base→canvas magnitude rescale, no
    // negation). The stabilizer gates each fresh alpha against the warp-aligned
    // previous, on the last-available flow.
    this.flow = null
    if (w.flow) {
      const b = cfg.baseRes, c = cfg.canvasRes
      const everyFrame = preset.skipFrames === 0
      const zeros = (h: number, wd: number) => this.backend.tensor(h, wd, 4, new Float32Array(h * wd * 4))
      // Flow stem input = the canvas frame (matting's input) downsampled to base,
      // matching training (x_base = interpolate(canvas → base)). A direct
      // source→base Input op is a single ~5× bilinear pass and aliases → garbles
      // the flow at edges.
      const curBaseDown = this.backend.ops.BilinearUpsample(networkInput.output, { outH: b.h, outW: b.w })
      const frameAHeld  = zeros(b.h, b.w)
      const net = new OpticalFlowNet(this.backend, frameAHeld, curBaseDown.output, tier.encoderTaps, w.flow, FLOW_DEC_W,
        cfg.flowFuseStem ? { fuseStem: true, halfTap: tier.halfTap } : {})
      const up = this.backend.ops.BilinearUpsample(net.output, { outH: c.h, outW: c.w })
      const flowScale = c.w / b.w   // backward flow (b→a): gather at p + flow (positive)

      const predBuf  = zeros(c.h, c.w)
      const stabPrev = zeros(c.h, c.w)
      const refWarp  = this.backend.ops.Warp(stabPrev, up.output, { flowScale })
      // Divergence finite-difference step ≈ canvas/flow ratio, so it spans ~1
      // base/4 pixel on the upsampled flow (where the occlusion seam lives).
      const stepX = Math.max(1, Math.round(c.w / net.output.w))
      const stepY = Math.max(1, Math.round(c.h / net.output.h))
      const stab     = this.backend.ops.Stabilize(up.output, predBuf, refWarp.output, stabPrev, { ...FLOW_STAB, stepX, stepY })

      // Skip tiers also warp the held last-inference alpha to make the skip-frame pred.
      const alphaHeld = everyFrame ? null : zeros(c.h, c.w)
      const predWarp  = alphaHeld ? this.backend.ops.Warp(alphaHeld, up.output, { flowScale }) : null

      this.flow = { tier, everyFrame, warm: false, networkInput, curBaseDown, frameAHeld, net, up,
                    predBuf, stabPrev, refWarp, stab, alphaHeld, predWarp }
    }

    this.preset      = preset
    this.skipCounter = 0
  }

  // Release everything: close any open background ports (worker-side
  // listeners stop firing) and destroy the backend (WebGPU device.destroy
  // releases all GPU buffers/textures/pipelines; WebGL loseContext does
  // the equivalent). After destroy(), this Renderer is unusable.
  destroy(): void {
    for (const port of this.bgVideoPorts.values()) port.close()
    this.bgVideoPorts.clear()
    this.backend.destroy()
  }

  getStats(): RendererStats {
    const now = performance.now()
    this.trimRecent(this.framesRenderedAt, now)
    this.trimSamplesAt(this.modelRunSamples,   now)
    this.trimSamplesAt(this.displayRunSamples, now)
    return {
      // window is 1s, so length = events-per-second
      fps:        this.framesRenderedAt.length,
      modelFps:   this.modelRunSamples.length,
      modelMs:    median(this.modelRunSamples.map(s => s.ms)),
      displayMs:  median(this.displayRunSamples.map(s => s.ms)),
      skipped:    this.skippedCount,
      preset:     this.preset?.model ?? 'boot',
      skipFrames: this.preset?.skipFrames ?? 0,
      enabled:    this.enabled,
      inputPath:  this.topology.input,
      outputPath: this.topology.output,
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  // Translate a canonical Background into a RenderOp effect config for `target`,
  // setting up that target's image/video input op as needed. Image/video bg
  // inputs are per-target so main and preview can show different backgrounds
  // without colliding. Sized to the (main) canvas — image/alpha/bg must share
  // h×w for the compositor. 'none' is a stub (the renderer composites
  // {passthrough} for 'none', never this config).
  private translateBackgroundFor(target: RenderTarget, bg: Background): RenderOpBackgroundConfig {
    // Switching this target away from video — close its prior port. Frames in
    // flight get dropped, which is fine.
    const prevPort = this.bgVideoPorts.get(target)
    if (bg.kind !== 'video' && prevPort) {
      prevPort.close()
      this.bgVideoPorts.delete(target)
    }

    switch (bg.kind) {
      case 'none':
        return { mode: 'solid', color: [0, 0, 0] }
      case 'transparent':
        return { mode: 'transparent' }
      case 'matte':
        return { mode: 'matte' }
      case 'color':
        return { mode: 'solid', color: bg.rgb }
      case 'blur':
        return { mode: 'blur', sigma: bg.sigma }
      case 'image': {
        const input = this.bgInputFor(this.bgImageInputs, target)
        input.setSource(bg.bitmap)
        input.run()
        return { mode: 'image', image: input.output }
      }
      case 'video': {
        const input = this.bgInputFor(this.bgVideoInputs, target)
        // Switching from one video to another on this target — close the prior.
        if (prevPort) prevPort.close()
        this.bgVideoPorts.set(target, bg.port)
        bg.port.onmessage = (e: MessageEvent<{ frame: VideoFrame }>) => {
          const frame = e.data.frame
          try {
            input.setSource(frame)
            input.run()
          } finally {
            frame.close()
          }
        }
        bg.port.start?.()
        return { mode: 'image', image: input.output }
      }
    }
  }

  // Get-or-create a per-target background Input op (canvas-sized).
  private bgInputFor(map: Map<RenderTarget, InputOp>, target: RenderTarget): InputOp {
    let input = map.get(target)
    if (!input) {
      input = this.backend.ops.Input(this.canvas.height, this.canvas.width)
      map.set(target, input)
    }
    return input
  }

  // Hard-cap helpers: bound buffer sizes so stale data can't grow without
  // bound between getStats() calls.
  private trim(arr: number[]): void {
    if (arr.length > 240) arr.splice(0, arr.length - 240)
  }
  private trimSamples(arr: { ts: number; ms: number }[]): void {
    if (arr.length > 240) arr.splice(0, arr.length - 240)
  }

  // Trim by time. Compares against the entry's ts field for sample arrays
  // and against the value itself for raw timestamp arrays.
  private trimRecent(arr: number[], now: number): void {
    const cutoff = now - STATS_WINDOW_MS
    let i = 0
    while (i < arr.length && arr[i] < cutoff) i++
    if (i > 0) arr.splice(0, i)
  }
  private trimSamplesAt(arr: { ts: number; ms: number }[], now: number): void {
    const cutoff = now - STATS_WINDOW_MS
    let i = 0
    while (i < arr.length && arr[i].ts < cutoff) i++
    if (i > 0) arr.splice(0, i)
  }
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const sorted = xs.slice().sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}
