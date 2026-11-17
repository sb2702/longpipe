// Typed control-plane bridge between main and worker.
//
// Request/response with UUIDs (matched on `request_id`); each sendMessage()
// returns a Promise that resolves with the worker's response, or REJECTS with
// a LongpipeError when the command threw in the worker ('control-failed'), no
// answer came in time ('control-timeout'), or the worker went away
// ('worker-crash'). No call can hang: every request has a timeout and
// terminate() rejects everything in flight.
//
// Persistent listeners stay registered across multiple events (used for
// errors / heartbeat / input-ended).
//
// The worker's own `error` / `messageerror` events (uncaught exception, a
// message that can't be deserialized, script load failure) are reported via
// `onCrash`; the Pipeline treats them as a worker crash and restarts.
//
// Data plane (frames, bitmaps) bypasses this: it flows through transferred
// streams or dedicated MessagePorts.

import type {
  CmdName,
  CmdDataMap,
  CmdResponseMap,
  EventName,
  EventMap,
  WorkerRequest,
  WorkerResponse,
  WorkerEvent,
} from './messages'
import { LongpipeError } from './errors'
import { TUNING } from './tuning'
import { createLogger } from './debug'

export type EventHandler<E extends EventName> = (data: EventMap[E]) => void

const log = createLogger('worker_controller')

interface Pending {
  cmd:     string
  resolve: (res: unknown) => void
  reject:  (err: LongpipeError) => void
  timer:   ReturnType<typeof setTimeout> | null
}

export interface SendOptions {
  // Override the default control timeout (ms). 0 = no timeout.
  timeoutMs?: number
}

export class WorkerController {
  private worker: Worker
  private pending: Map<string, Pending> = new Map()
  private persistent: Map<string, (data: unknown) => void> = new Map()
  private terminated = false

  constructor(worker: Worker, private onCrash?: (message: string, cause?: unknown) => void) {
    this.worker = worker
    this.worker.addEventListener('message', this.handleMessage.bind(this))
    this.worker.addEventListener('error', (e: ErrorEvent) => {
      // Keep the default console report; just tell the pipeline.
      this.onCrash?.(`worker error: ${e.message || 'uncaught exception / script failed to load'}`, e)
    })
    this.worker.addEventListener('messageerror', (e) => {
      this.onCrash?.('worker messageerror: a message could not be deserialized', e)
    })
    log('constructed; listeners attached')
  }

  get isTerminated(): boolean { return this.terminated }

  private handleMessage(event: MessageEvent<WorkerResponse | WorkerEvent>): void {
    const { request_id } = event.data
    const persistent = this.persistent.get(request_id)
    if (persistent) { persistent((event.data as WorkerEvent).res); return }
    const p = this.pending.get(request_id)
    if (!p) return
    this.pending.delete(request_id)
    if (p.timer) clearTimeout(p.timer)
    const msg = event.data as WorkerResponse
    if (msg.error) p.reject(new LongpipeError('control-failed', msg.error.message, { cause: msg.error }))
    else p.resolve(msg.res)
  }

  addPersistentListener<E extends EventName>(name: E, handler: EventHandler<E>): void {
    this.persistent.set(name, handler as (data: unknown) => void)
  }

  removePersistentListener(name: EventName): void {
    this.persistent.delete(name)
  }

  sendMessage<C extends CmdName>(
    cmd:      C,
    data:     CmdDataMap[C],
    transfer: Transferable[] = [],
    opts:     SendOptions = {},
  ): Promise<CmdResponseMap[C]> {
    if (this.terminated) {
      return Promise.reject(new LongpipeError('worker-crash', `${cmd}: worker is not running`))
    }
    const request_id = crypto.randomUUID()
    log('sendMessage:', cmd, 'request_id=', request_id, 'transferables=', transfer.length)
    return new Promise((resolve, reject) => {
      const timeoutMs = opts.timeoutMs ?? TUNING.controlTimeoutMs
      const timer = timeoutMs > 0
        ? setTimeout(() => {
            if (!this.pending.delete(request_id)) return
            reject(new LongpipeError('control-timeout', `${cmd}: no response from the worker after ${timeoutMs} ms`))
          }, timeoutMs)
        : null
      this.pending.set(request_id, {
        cmd,
        resolve: resolve as (res: unknown) => void,
        reject,
        timer,
      })
      const message: WorkerRequest<C> = { cmd, data, request_id }
      try {
        this.worker.postMessage(message, transfer)
      } catch (err) {
        // DataCloneError etc. — fail this call, not the pipeline.
        this.pending.delete(request_id)
        if (timer) clearTimeout(timer)
        reject(new LongpipeError('control-failed', `${cmd}: could not send to the worker: ${(err as Error).message}`, { cause: err }))
      }
    })
  }

  // Terminate the worker and reject every in-flight request with
  // 'worker-crash' (the caller is restarting or destroying).
  //
  // graceMs > 0: the worker first gets a 'destroy' command so it can release
  // the camera frames it holds (pipe, pending slot, stream queues) and is
  // killed once it answers, or after graceMs (hung / already dead). Killing it
  // outright destroys those VideoFrames unclosed; Chromium then reports "A
  // VideoFrame was garbage collected without being closed" in the PAGE's
  // console (a frame keeps the close auditor of the context that created it).
  // Either way the controller is finished at once: nothing it does in the
  // grace period reaches the pipeline.
  terminate(reason = 'worker terminated', graceMs = 0): void {
    if (this.terminated) return
    this.terminated = true
    if (graceMs > 0) this.retire(graceMs)
    else this.worker.terminate()
    for (const [, p] of this.pending) {
      if (p.timer) clearTimeout(p.timer)
      p.reject(new LongpipeError('worker-crash', `${p.cmd}: ${reason}`))
    }
    this.pending.clear()
    this.persistent.clear()
  }

  private retire(graceMs: number): void {
    const worker = this.worker
    const request_id = crypto.randomUUID()
    let done = false
    const kill = () => {
      if (done) return
      done = true
      clearTimeout(timer)
      worker.removeEventListener('message', onMessage)
      worker.terminate()
    }
    const onMessage = (e: MessageEvent<{ request_id?: string }>) => { if (e.data?.request_id === request_id) kill() }
    const timer = setTimeout(kill, graceMs)
    worker.addEventListener('message', onMessage)
    try {
      const message: WorkerRequest<'destroy'> = { cmd: 'destroy', data: {} as Record<string, never>, request_id }
      worker.postMessage(message)
    } catch {
      kill()
    }
  }
}
