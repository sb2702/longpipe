// Worklet processor source string, injected at build time by tsup's
// inline-worklet plugin (mirrors pipeline/worker_inline.ts). Empty in source so
// dev mode (vite serving src/) falls back to the URL-based addModule — which
// works in dev because src/ is same-origin. In published builds the plugin
// replaces this with the bundled processor source, which the denoiser wraps in
// a Blob URL and audioWorklet.addModule()s — same-origin from any CDN consumer.
export const WORKLET_SOURCE = ''
