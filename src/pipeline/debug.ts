// Centralized log gate for the SDK. Both main-thread and worker each run in
// their own JS context with their own copy of `enabled`, so the worker side
// has its flag set independently when init data arrives.
//
// Usage:
//   const log = createLogger('pipeline')
//   log('hello', 42)         → '[longpipe/pipeline] hello 42' (if enabled)
//   log.warn('something off') → '[longpipe/pipeline] something off' via console.warn
//
// console.error is intentionally NOT gated — genuine errors should always
// surface, regardless of the debug flag.

let enabled = false

export function setDebug(value: boolean): void {
  enabled = value
}

export function isDebug(): boolean {
  return enabled
}

export type Logger = {
  (...args: unknown[]): void
  warn(...args: unknown[]): void
}

export function createLogger(scope: string): Logger {
  const prefix = `[longpipe/${scope}]`
  const fn = ((...args: unknown[]) => {
    if (enabled) console.log(prefix, ...args)
  }) as Logger
  fn.warn = (...args: unknown[]) => {
    if (enabled) console.warn(prefix, ...args)
  }
  return fn
}
