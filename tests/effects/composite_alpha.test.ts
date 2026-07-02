import { describe, it, expect } from 'vitest'
import { WebGLBackend } from '~/model/backends/webgl/index'
import { WebGPUBackend } from '~/model/backends/webgpu/index'

// The transparent + matte compositors render straight to the canvas (no Tensor
// output), so — unlike the op tests — we assert on canvas pixels. WebGL is the
// correctness path (gl.readPixels off the default framebuffer). WebGPU gets a
// compile-and-run smoke (its swapchain isn't COPY_SRC-configured, so pixel
// readback would need backend changes; the shader compiling + a full frame
// submitting is what we're guarding).
//
// Two horizontal pixels: x=0 is fully opaque foreground, x=1 is fully keyed
// out (alpha 0). Both backends' canvases are premultiplied, so the expected
// output is premultiplied: fg·a in rgb, matte in a.

const RED_THEN_BLUE = new Float32Array([
  1, 0, 0, 1,   // px0: red,  image alpha unused by the compositor
  0, 0, 1, 1,   // px1: blue
])
const ALPHA_1_THEN_0 = new Float32Array([
  1, 0, 0, 0,   // px0: matte = 1  (kept)
  0, 0, 0, 0,   // px1: matte = 0  (transparent)
])

function readWebGLCanvas(backend: WebGLBackend, w: number, h: number): Uint8Array {
  const gl = backend.gl
  const buf = new Uint8Array(w * h * 4)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null)     // default framebuffer === canvas
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf)
  return buf
}

describe('CompositeTransparent (WebGL)', () => {
  it('writes premultiplied subject with the matte as alpha', () => {
    const backend = WebGLBackend.create({ canvas: new OffscreenCanvas(2, 1) })
    const image = backend.tensor(1, 2, 4, RED_THEN_BLUE)
    const alpha = backend.tensor(1, 2, 4, ALPHA_1_THEN_0)

    backend.presenters.CompositeTransparent(image, alpha).run()
    const px = readWebGLCanvas(backend, 2, 1)
    backend.destroy()

    // px0: red × 1 → (255,0,0,255). px1: keyed out → (0,0,0,0).
    expect(Array.from(px.slice(0, 4))).toEqual([255, 0, 0, 255])
    expect(Array.from(px.slice(4, 8))).toEqual([0, 0, 0, 0])
  })
})

describe('CompositeMatte (WebGL)', () => {
  it('writes a premultiplied white silhouette (rgb = a, alpha = a)', () => {
    const backend = WebGLBackend.create({ canvas: new OffscreenCanvas(2, 1) })
    const alpha = backend.tensor(1, 2, 4, ALPHA_1_THEN_0)

    backend.presenters.CompositeMatte(alpha).run()
    const px = readWebGLCanvas(backend, 2, 1)
    backend.destroy()

    expect(Array.from(px.slice(0, 4))).toEqual([255, 255, 255, 255])
    expect(Array.from(px.slice(4, 8))).toEqual([0, 0, 0, 0])
  })
})

describe('CompositeTransparent / CompositeMatte (WebGPU)', () => {
  it('compiles and submits a frame', async () => {
    if (!navigator.gpu || !(await navigator.gpu.requestAdapter().catch(() => null))) {
      // No WebGPU in this browser — the WebGL cases above cover correctness.
      return
    }
    const backend = await WebGPUBackend.create({ canvas: new OffscreenCanvas(2, 1) })
    const image = backend.tensor(1, 2, 4, RED_THEN_BLUE)
    const alpha = backend.tensor(1, 2, 4, ALPHA_1_THEN_0)

    // Throws if the WGSL fails to compile or the pass fails to submit.
    expect(() => backend.presenters.CompositeTransparent(image, alpha).run()).not.toThrow()
    expect(() => backend.presenters.CompositeMatte(alpha).run()).not.toThrow()
    await backend.sync()
    backend.destroy()
  })
})
