// Worker source string injected at build time by tsup's inline-worker plugin.
// Empty in source so dev mode (vite serving from src/) falls back to the
// URL-based worker load — which works in dev because src/ is same-origin.
// In published builds the plugin replaces this string with the bundled worker
// source so consumers can spawn the worker via Blob URL — same-origin no
// matter where the SDK was loaded from (npm + bundler, esm.sh, jsdelivr, etc).
export const WORKER_SOURCE = ''
