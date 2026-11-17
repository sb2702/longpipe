// Output relay worker for the MSTG (Chromium) output path.
//
// Why it exists: the generator's `writable` can be transferred exactly once.
// In 0.0.18 it went straight to the GPU worker, so when that worker died the
// output track froze forever (it stays `live`, but nothing can write to it).
// Now the writable lives in this tiny worker, which never touches the GPU and
// so has nothing that can crash. It writes ONE of two sources into the
// generator:
//
//   'raw'    — a MediaStreamTrackProcessor readable on the camera track
//              (startup before the GPU worker delivers, no-GPU, GPU loss,
//              GPU-worker crash/restart)
//   'worker' — VideoFrames the GPU worker posts on a MessagePort (normal)
//
// Handoff raw → worker happens after the worker's HANDOFF_AT_FRAME-th frame
// (its first frames are slow while shaders warm up), exactly like the
// bitmap-shuttle main-thread passthrough.
//
// Happy-path cost vs. 0.0.18: one extra worker→worker postMessage per frame
// (VideoFrame transferred, zero-copy). Measured in headless Chromium 147,
// 1280×720 @ 30 fps: p50 latency 0.2 ms → 0.3–0.4 ms, no main-thread work,
// unaffected by main-thread jank (a main-thread relay was p95 60 ms under a
// 60 ms/200 ms jank loop). End-to-end A/B vs 0.0.18 (1280×720 @ 30 fps,
// tier small): identical output fps and frame-interval p50/p95.
//
// Kept as a plain JS string (no imports, no bundler helpers) so it can be
// started from a Blob URL identically in dev and in published builds.

export const RELAY_SOURCE = `
'use strict';
const HANDOFF_AT_FRAME = 3;
let writer = null;
let busy = false;
let raw = null;        // reader on the camera (kept across raw ↔ worker switches)
let rawOn = false;     // raw frames go to the output (else they're dropped)
let port = null;       // active GPU-worker frame port
let workerFrames = 0;
let framesOut = 0;

function write(frame) {
  // Drop instead of queueing: a late frame is worse than a skipped one.
  if (!writer || busy) { frame.close(); return; }
  busy = true;
  framesOut++;
  writer.write(frame).then(() => { busy = false; }, () => { busy = false; });
}

function stopReader() {
  rawOn = false;
  if (!raw) return;
  const r = raw; raw = null;
  r.cancel().catch(() => {});
}

// One reader per camera track for the relay's lifetime. It keeps reading (and
// closing) while the GPU worker owns the output: cancelling a transferred
// camera readable and transferring a fresh one on every restart is what left
// unclosed VideoFrames behind in the page (Chromium), and an unread reader
// would pin camera buffers.
async function pump(reader) {
  for (;;) {
    let res;
    try { res = await reader.read(); } catch (e) { break; }
    if (res.done) break;
    if (raw === reader && rawOn) write(res.value);
    else res.value.close();
    if (raw !== reader) break;
  }
  if (raw === reader) { raw = null; rawOn = false; }
}

function startRaw(readable) {
  if (readable) {
    stopReader();
    raw = readable.getReader();
    pump(raw);
  }
  rawOn = !!raw;
  workerFrames = 0;
  postMessage({ type: 'mode', mode: 'raw' });
}

function detach() {
  if (port) { port.onmessage = null; port.close(); port = null; }
}

onmessage = (e) => {
  const d = e.data;
  if (d.type === 'init') {
    writer = d.writable.getWriter();
  } else if (d.type === 'raw') {
    if (d.detach) detach();
    startRaw(d.readable);
  } else if (d.type === 'worker') {
    detach();
    const p = d.port;
    port = p;
    workerFrames = 0;
    p.onmessage = (ev) => {
      const frame = ev.data.frame;
      if (port !== p) { frame.close(); return; }
      if (rawOn) {
        if (++workerFrames < HANDOFF_AT_FRAME) { frame.close(); return; }
        rawOn = false;
        postMessage({ type: 'mode', mode: 'worker' });
      }
      write(frame);
    };
    p.start();
  } else if (d.type === 'stats') {
    postMessage({ type: 'stats', framesOut, mode: rawOn ? 'raw' : (port ? 'worker' : 'idle') });
  } else if (d.type === 'close') {
    stopReader();
    detach();
    if (writer) writer.close().catch(() => {});
    writer = null;
    postMessage({ type: 'closed' });
  }
};
`

export function createRelayWorker(): Worker {
  const url = URL.createObjectURL(new Blob([RELAY_SOURCE], { type: 'application/javascript' }))
  const w = new Worker(url)
  URL.revokeObjectURL(url)
  return w
}
