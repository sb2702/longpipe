import { describe, it, expect } from 'vitest'
import { resolveModel, validateModelOptions } from '~/pipeline/model_choice'
import { LongpipeError } from '~/pipeline/errors'

describe('model option', () => {
  it("defaults to 'longpipe'", () => {
    expect(resolveModel(undefined, { w: 1280, h: 720 })).toEqual({ family: 'longpipe' })
  })

  it('picks the MediaPipe variant from the output aspect, or honours a forced one', () => {
    expect(resolveModel('mediapipe-selfie', { w: 1280, h: 720 })).toEqual({ family: 'mediapipe-selfie', variant: 'landscape' })
    expect(resolveModel('mediapipe-selfie', { w: 640, h: 480 })).toEqual({ family: 'mediapipe-selfie', variant: 'square' })
    expect(resolveModel('mediapipe-selfie-square', { w: 1280, h: 720 })).toEqual({ family: 'mediapipe-selfie', variant: 'square' })
    expect(resolveModel('mediapipe-selfie-landscape', { w: 480, h: 480 })).toEqual({ family: 'mediapipe-selfie', variant: 'landscape' })
  })

  it('rejects unknown models', () => {
    expect(() => resolveModel('mediapipe' as any, { w: 1, h: 1 })).toThrow(/unknown model/)
  })

  it('Longpipe-only options throw under a MediaPipe model', () => {
    const msg = /requires a Longpipe model; set model: 'longpipe'/
    expect(() => validateModelOptions('mediapipe-selfie', { touchup: {} })).toThrow(msg)
    expect(() => validateModelOptions('mediapipe-selfie', { reframe: true })).toThrow(msg)
    expect(() => validateModelOptions('mediapipe-selfie', { preset: 'fast' })).toThrow(msg)
    expect(() => validateModelOptions('mediapipe-selfie', { preset: 'auto', reframe: false })).not.toThrow()
    expect(() => validateModelOptions('longpipe', { touchup: {}, reframe: true, preset: 'fast' })).not.toThrow()
    expect(() => validateModelOptions(undefined, { touchup: {} })).not.toThrow()
    expect(() => validateModelOptions('mediapipe-selfie', { touchup: {} })).toThrow(LongpipeError)
  })
})
