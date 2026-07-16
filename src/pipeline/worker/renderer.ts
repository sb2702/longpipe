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
import { FaceHeatmapNet } from '~/model/networks/face_heatmap_net.ts'
import { LandmarkNet } from '~/model/networks/landmark_net.ts'
import { TIER_CONFIG } from '~/model/tier_config.ts'
import type { FaceTopology, FaceTouchupParams, FaceTouchupStageOp } from '~/model/backend.ts'

const FLOW_DEC_W = 16   // shipping flow-head width
// Flow-gated stabilizer params. tLo/tHi are flow magnitudes in BASE-res pixels
// (the flow predicts base-res-unit displacements); envelope peak-hold mirrors the
// eval harness's known-good (leak 0.2, release 0.9). Tunable.
// tDiv/divScale: the occlusion-seam term — the gate also opens where |div(flow)|
// exceeds tDiv (a boundary the magnitude gate is blind to, since the revealed
// background is static). stepX/stepY are added per-tier in setPreset (≈ canvas/flow
// resolution ratio so the finite-difference spans one base/4 pixel). Tunable.
const FLOW_STAB = { tLo: 0.15, tHi: 2.5, leak: 0.15, release: 0.9, tDiv:1.0, divScale: 2.0 }
// Landmark crop config — mirrors landmark training (256² crop, ImageNet norm).
const LM_CROP = 256
const IMAGENET_MEAN: [number, number, number] = [0.485, 0.456, 0.406]
const IMAGENET_STD:  [number, number, number] = [0.229, 0.224, 0.225]
const FACE_BOX_DEFAULTS = { win: 3, thresh: 0.15, boxScale: 2.4 }
// Multi-face decode + retouch. Gated to medium and above: xs/small decode at a
// 32×20 / 48×28 grid, too coarse to resolve a second face reliably, so they keep
// the single-face path (FaceBoxFromHeatmaps + 1 slot) — which the K=1 atlas
// layout leaves byte-identical to the pre-multi-face behavior.
const MULTI_FACE_TIERS = new Set(['medium', 'large', 'xl'])
const MULTI_FACE_K = 4
// Keypoint-match radius as a fraction of interocular distance, for the eye-pair
// grouping. Prototyped in demo/face.ts; 0.6 is the default the probe settled on.
const FACE_GROUP_TOL = 0.6
// Occupancy probe cadence, in face-chain frames. The box tensor is 64 bytes, but
// the two backends charge very differently for reading it: WebGPU's readback is a
// genuine async buffer map (cheap, off the critical path), while WebGL's is a
// blocking gl.readPixels that flushes the pipeline — `async` in signature only.
// So WebGL probes rarely enough to amortize the stall; WebGPU can afford to be
// responsive. The cost of a slow probe is only latency on a face ARRIVING
// (it goes unretouched until the next probe), never a wrong-looking frame.
const FACE_PROBE_INTERVAL = { webgpu: 6, webgl: 30 }
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

// Face-effect configuration (touch-up today; AR later). The landmark model is
// SEPARATE weights (one model, all tiers); topology/weight-mask are static
// assets. box tunes the heatmap→crop decode (defaults from the probe pages).
export interface FaceEffectsConfig {
  landmarkWeights: ArrayBuffer
  topology:        FaceTopology
  touchup:         FaceTouchupParams
  box?:            { win?: number; thresh?: number; boxScale?: number }
}

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
  // Face-effect chain — null unless setFaceEffects() configured it AND the
  // loaded .bin carries a `face` blob. The heatmaps are a temporal citizen
  // exactly like alpha: FaceHeatmapNet runs on matting-inference frames (its
  // taps are only fresh then); on skip frames the HELD heatmaps are warped
  // forward with the same flow field (rescaled to heatmap res). Landmarks run
  // EVERY frame from the current (live or warped) heatmaps — the regressor is
  // tiny and crop-jitter training tolerates the box.
  //   hmBuf     : current heatmaps (live or warped) — the box op's input
  //   hmHeld    : warp source, copied on inference frames (flow skip tiers only)
  //   boxStable : box carrier (1×K×4) — crops + effect stage bind this tensor
  //   crops/lms : one per face slot, differing ONLY by the box slot they read
  //   packLm    : ChannelConcat tree → one buffer holding all K faces' landmarks,
  //               which the touch-up mesh vertex shader indexes per instance
  //               (null at K=1 — lms[0].output is already that buffer)
  private faceChain: {
    face: FaceHeatmapNet
    hmBuf: Tensor
    hmHeld: Tensor | null
    hmFlowDown: Op | null
    hmWarp: Op | null
    boxOp: Op
    boxStable: Tensor
    crops: Op[]
    lms: LandmarkNet[]
    packLm: Op[]
    stage: FaceTouchupStageOp
  } | null = null
  private faceCfg: FaceEffectsConfig | null = null
  // Occupancy gating. FaceBoxesFromHeatmaps fills slots 0..n-1 and zeroes the
  // rest, so live faces are always a PREFIX — one count is enough, no per-slot
  // mask. faceLive is how many crops/landmark nets run AND how many mesh
  // instances the touch-up draws; the two must not diverge (stale landmarks under
  // a live box smear the old face's mesh onto the new one). Starts at K so the
  // frames before the first probe are correct rather than cheap.
  private faceLive:         number  = 0
  private faceProbePending: boolean = false
  private faceProbeCounter: number  = 0
  // Pieces held for face-chain (re)builds outside setPreset.
  private tier: TierModel | null = null
  private tierInput: InputOp | null = null
  private lastTierWeights: any = null

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
    const faceActive    = this.enabled && hasNet && this.faceChain !== null
    const mainEffect    = this.enabled && hasNet && this.currentBackground.kind !== 'none'
    const needsModel    = mainEffect || faceActive   // face rides the encoder → needs the matting pass
    const previewTick   = this.isPreviewTick(now)
    const previewEffect = previewTick && hasNet && this.previewBg !== null && this.previewBg.kind !== 'none'


    // Model gate. Flow tiers run the matting + flow + stabilizer in stepFlow;
    // non-flow tiers follow the plain skip cadence. A preview tick that needs alpha
    // forces a run when main didn't already produce one this frame.
    let ranModel = false
    if (needsModel && this.flow) {
      ranModel = this.stepFlow()
    } else if (needsModel) {
      if (this.shouldRunModel()) { this.runModelOnce(); ranModel = true }
      else this.skippedCount++
    }
    if (!ranModel && previewEffect) { this.runModelOnce(); ranModel = true }

    // Face chain: heatmaps (live on inference frames / warped on skips) → box →
    // crop → landmarks. Runs every frame while active; the touch-up stage itself
    // runs inside the RenderOp effect chain during the composite below.
    if (faceActive) this.stepFace(ranModel)

    // Composite. The display input (full-res image, shared by both surfaces) is
    // refreshed once inside whichever composite path runs first.
    const td = performance.now()
    if (previewTick && this.previewCanvas) {
      if (this.backendKind === 'webgl') {
        // WebGL: one canvas — preview composited + snapshotted, then main last
        // so the output adapter captures main. All synchronous (no race).
        this.compositePreviewWebGL(mainEffect, previewEffect, faceActive)
      } else {
        // WebGPU: main + preview render to independent swapchains; order free.
        this.compositeMain(mainEffect, faceActive)
        this.renderOp.compositeTo('preview', this.previewSpec(previewEffect))
      }
      this.lastPreviewAt = now
    } else {
      this.compositeMain(mainEffect, faceActive)
    }
    this.displayRunSamples.push({ ts: performance.now(), ms: performance.now() - td })
    this.trimSamples(this.displayRunSamples)

    this.framesRenderedAt.push(performance.now())
    this.trim(this.framesRenderedAt)
  }

  // Main surface: background effect (runDisplay), effect-chain-only output
  // (fg-passthrough — touch-up on, background 'none'), or raw passthrough.
  // All refresh the shared display input as their first step.
  private compositeMain(effect: boolean, faceActive: boolean): void {
    if (effect)          this.renderOp.runDisplay()
    else if (faceActive) this.renderOp.runFgPassthrough()
    else                 this.renderOp.runPassthrough()
  }

  // WebGL preview present: render preview to the single canvas, snapshot it to
  // the preview canvas's bitmaprenderer context, then re-composite main so the
  // canvas rests on main content. transferToImageBitmap detaches the canvas
  // backing (left blank), but main is recomposited before this synchronous
  // block ends, so neither the output adapter nor a continuous captureStream
  // ever observes preview/blank content.
  private compositePreviewWebGL(mainEffect: boolean, previewEffect: boolean, faceActive: boolean): void {
    this.renderOp.refreshDisplayInput()
    this.renderOp.runEffects()   // chain feeds both surfaces' compositors
    this.renderOp.compositeTo('preview', this.previewSpec(previewEffect))
    const bmp = this.canvas.transferToImageBitmap()   // this.canvas === backend.canvas, typed OffscreenCanvas
    this.previewCtx!.transferFromImageBitmap(bmp)
    // main last; effect / chain-only / raw per the main-surface state
    this.renderOp.compositeMain(mainEffect ? 'effect' : faceActive ? 'fg' : 'raw')
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

  // One face-chain frame. On matting-inference frames the heatmaps are fresh
  // (taps just updated); on skip frames the held heatmaps are warped forward
  // with this frame's flow (stepFlow already ran the flow net on skip frames).
  // Without a flow blob the heatmaps simply hold between inference frames.
  private stepFace(ranModel: boolean): void {
    const fc = this.faceChain!
    if (ranModel) {
      fc.face.run()
      this.backend.copyTensor(fc.face.output, fc.hmBuf)
      if (fc.hmHeld) this.backend.copyTensor(fc.face.output, fc.hmHeld)
    } else if (fc.hmWarp && this.flow) {
      fc.hmFlowDown!.run()   // flow → heatmap res (values rescaled via flowScale)
      fc.hmWarp.run()
      this.backend.copyTensor(fc.hmWarp.output, fc.hmBuf)
    }
    // else: hold the last heatmaps (non-flow tier skip frame)

    fc.boxOp.run()
    this.backend.copyTensor(fc.boxOp.output, fc.boxStable)

    // Occupancy gating: run a crop + landmark net only for slots that hold a
    // face. LandmarkNet is 13 dispatches at 256², so running all K every frame
    // cost ~4× for the common single-face case. The box op still decodes all K
    // slots every frame (one dispatch) — only the per-face model runs are gated,
    // so the probe below sees the true count regardless of what we skipped.
    this.probeFaceOccupancy(fc)
    const live = fc.crops.length === 1 ? 1 : this.faceLive
    for (let i = 0; i < live; i++) { fc.crops[i].run(); fc.lms[i].run() }
    if (live > 0) for (const p of fc.packLm) p.run()
    fc.stage.setActiveSlots(live)
  }

  // Refresh faceLive from the box tensor, throttled and never overlapping. A
  // face ARRIVING is unretouched until the next probe (≤100ms WebGPU, ≤500ms
  // WebGL); a face LEAVING keeps its slot running until then, which costs a
  // landmark run and looks identical (its box score is 0, so nothing draws).
  private probeFaceOccupancy(fc: NonNullable<Renderer['faceChain']>): void {
    const K = fc.crops.length
    if (K === 1) return
    const interval = FACE_PROBE_INTERVAL[this.backendKind]
    if (this.faceProbePending || this.faceProbeCounter++ % interval !== 0) return
    this.faceProbePending = true
    this.backend.readback(fc.boxStable).then(b => {
      let n = 0
      while (n < K && b[n * 4 + 3] > 0) n++
      this.faceLive = n
      this.faceProbePending = false
    }).catch(() => { this.faceProbePending = false })   // shutdown races are expected
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

    this.preset          = preset
    this.skipCounter     = 0
    this.tier            = tier
    this.tierInput       = networkInput
    this.lastTierWeights = w
    this.buildFaceChain()
  }

  // Configure (or clear) the face-effect chain. Takes effect immediately when a
  // preset is attached; otherwise applied by the next setPreset. Passing a new
  // config rebuilds the chain in place (adaptive tier swaps do the same via
  // setPreset — the chain rides the current tier's encoder taps).
  setFaceEffects(cfg: FaceEffectsConfig | null): void {
    this.faceCfg = cfg
    if (this.preset) this.buildFaceChain()
  }

  // (Re)build the face chain + RenderOp effect chain from faceCfg and the
  // current tier. No-ops (clearing both) when unconfigured or the .bin has no
  // face blob.
  private buildFaceChain(): void {
    this.faceChain = null
    this.renderOp.setEffectChain([])
    const cfg = this.faceCfg
    const w = this.lastTierWeights
    if (!cfg || !this.tier || !this.tierInput || !this.preset) return
    if (!w?.face) {
      // Loud, not silent: a missing face blob means the tier .bin predates the
      // face-head export — touch-up cannot run. (Cost a debugging marathon once.)
      console.warn(`[longpipe/renderer] touchup configured but the '${this.preset.model}' weights have no face blob — effect disabled. Re-export the tier with a face-trained checkpoint.`)
      return
    }

    const face = new FaceHeatmapNet(this.backend, this.tier.encoderTaps, w.face)
    const hm = face.output
    const zerosT = (h: number, wd: number, c: number) =>
      this.backend.tensor(h, wd, c, new Float32Array(h * wd * c))
    const hmBuf = zerosT(hm.h, hm.w, hm.c)

    // Heatmap warp carriers — only meaningful on flow tiers that skip frames
    // (every-frame tiers refresh the heatmaps each frame; non-flow tiers hold).
    let hmHeld: Tensor | null = null
    let hmFlowDown: Op | null = null
    let hmWarp: Op | null = null
    if (this.flow && (this.preset.skipFrames ?? 0) > 0) {
      const baseW = TIER_CONFIG[this.preset.model].baseRes.w
      hmHeld = zerosT(hm.h, hm.w, hm.c)
      hmFlowDown = this.backend.ops.BilinearUpsample(this.flow.net.output, { outH: hm.h, outW: hm.w })
      // flow values are base-res px → heatmap-res px
      hmWarp = this.backend.ops.Warp(hmHeld, hmFlowDown.output, { flowScale: hm.w / baseW })
    }

    // K faces on medium+, 1 on xs/small (grid too coarse — see MULTI_FACE_TIERS).
    const K = MULTI_FACE_TIERS.has(this.preset.model) ? MULTI_FACE_K : 1
    const boxCfg = { ...FACE_BOX_DEFAULTS, ...(cfg.box ?? {}) }
    const boxOp = K > 1
      ? this.backend.ops.FaceBoxesFromHeatmaps(hmBuf, { ...boxCfg, maxFaces: K, tol: FACE_GROUP_TOL })
      : this.backend.ops.FaceBoxFromHeatmaps(hmBuf, boxCfg)
    const boxStable = zerosT(1, K, 4)

    // One crop + landmark net per slot; only the box slot differs.
    const lmWeights = loadWeightsFromBinary(cfg.landmarkWeights) as any
    const cropSrc = this.tierInput.output   // narrowing is lost inside the closure
    const crops = Array.from({ length: K }, (_, i) =>
      this.backend.ops.CropResample(cropSrc, boxStable, {
        outH: LM_CROP, outW: LM_CROP, mean: IMAGENET_MEAN, std: IMAGENET_STD, slot: i,
      }))
    const lms = crops.map(c => new LandmarkNet(this.backend, c.output, lmWeights))

    // The touch-up mesh vertex shader indexes ONE landmark buffer by instance, so
    // the K outputs are concatenated into one (each is 1×1×956; ChannelConcat is
    // channel-generic and its 1×1 output layout is exactly face·239 + i/2).
    // Balanced tree so K=4 is two concats deep rather than three.
    const packLm: Op[] = []
    let lmPacked: Tensor = lms[0].output
    if (K > 1) {
      let level: Tensor[] = lms.map(l => l.output)
      while (level.length > 1) {
        const next: Tensor[] = []
        for (let i = 0; i < level.length; i += 2) {
          const op = this.backend.ops.ChannelConcat(level[i], level[i + 1])
          packLm.push(op)
          next.push(op.output)
        }
        level = next
      }
      lmPacked = level[0]
    }

    const stage = this.backend.ops.FaceTouchupStage(
      this.renderOp.displayImage, lmPacked, boxStable, cfg.topology, { ...cfg.touchup, slots: K })
    this.renderOp.setEffectChain([stage])
    // Correct-before-cheap until the first probe lands: assume every slot holds
    // a face, and let the probe settle it down within a few frames.
    this.faceLive = K
    this.faceProbeCounter = 0
    this.faceChain = { face, hmBuf, hmHeld, hmFlowDown, hmWarp, boxOp, boxStable, crops, lms, packLm, stage }
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
