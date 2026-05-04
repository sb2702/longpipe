// IEEE 754 binary16 (fp16) conversion helpers. Used by both WebGPU and WebGL
// backends when running in dtype: 'f16' — weights ship as raw fp16 bits in a
// Uint16Array, and the backend either stores them directly (WebGPU storage
// buffer / WebGL RGBA16F texture) or converts back for readback.
//
// Round-to-nearest-even, with proper subnormal / inf / NaN handling.

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

export function floatToHalf(val: number): number {
  f32[0] = val
  const x = u32[0]

  const sign = (x >>> 16) & 0x8000
  // Strip sign and rebias exponent from f32 (bias 127) to f16 (bias 15).
  const mantissaAndExp = x & 0x7fffffff
  const exp = (mantissaAndExp >>> 23) - 127 + 15

  // NaN / Inf
  if ((x & 0x7f800000) === 0x7f800000) {
    if ((x & 0x007fffff) !== 0) return sign | 0x7e00          // NaN (preserve a payload bit)
    return sign | 0x7c00                                      // ±Inf
  }

  // Overflow → ±Inf
  if (exp >= 31) return sign | 0x7c00

  // Subnormal / underflow
  if (exp <= 0) {
    if (exp < -10) return sign                                // too small → ±0
    // Build subnormal: shift the (implicit-1 | mantissa) right by (1 - exp).
    const m = (mantissaAndExp & 0x007fffff) | 0x00800000
    const shift = 14 - exp                                     // 14 = 23 - 10 + 1
    let half = m >>> shift
    // Round-to-nearest-even on the dropped bits.
    const round = (m >>> (shift - 1)) & 1
    const sticky = (m & ((1 << (shift - 1)) - 1)) !== 0 ? 1 : 0
    if (round && (sticky || (half & 1))) half += 1
    return sign | half
  }

  // Normal range
  let mant = (mantissaAndExp >>> 13) & 0x03ff
  const round = (mantissaAndExp >>> 12) & 1
  const sticky = (mantissaAndExp & 0x00000fff) !== 0 ? 1 : 0
  let halfBits = (exp << 10) | mant
  if (round && (sticky || (mant & 1))) {
    halfBits += 1                                              // mantissa carry into exp handled by addition
    if (((halfBits >>> 10) & 0x1f) === 31) return sign | 0x7c00 // overflow to inf after rounding
  }
  return sign | halfBits
}

export function halfToFloat(h: number): number {
  const sign = (h & 0x8000) << 16
  const exp  = (h & 0x7c00) >>> 10
  const mant = h & 0x03ff

  if (exp === 0) {
    if (mant === 0) { u32[0] = sign; return f32[0] }           // ±0
    // Subnormal — normalise.
    let m = mant, e = 1
    while ((m & 0x0400) === 0) { m <<= 1; e -= 1 }
    m &= 0x03ff
    u32[0] = sign | ((e + 127 - 15) << 23) | (m << 13)
    return f32[0]
  }
  if (exp === 31) {
    u32[0] = sign | 0x7f800000 | (mant << 13)                  // ±Inf or NaN
    return f32[0]
  }
  u32[0] = sign | ((exp + 127 - 15) << 23) | (mant << 13)
  return f32[0]
}

export function float32ArrayToHalf(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = floatToHalf(src[i])
  return out
}

export function halfArrayToFloat32(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length)
  for (let i = 0; i < src.length; i++) out[i] = halfToFloat(src[i])
  return out
}
