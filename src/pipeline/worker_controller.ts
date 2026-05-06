// Typed control-plane bridge between main and worker.
//
// Request/response with UUIDs (matched on `request_id`); each sendMessage()
// returns a Promise that resolves with the worker's response. Persistent
// listeners stay registered across multiple events (used for stats /
// errors / ready).
//
// Data plane (frames, bitmaps) bypasses this — flows through transferred
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

import { createLogger } from './debug'

export type EventHandler<E extends EventName> = (data: EventMap[E]) => void

const log = createLogger('worker_controller')

export class WorkerController {
  private worker: Worker
  private listeners: Map<string, (data: unknown) => void> = new Map()
  private persistentEvents: Set<string> = new Set()

  constructor(worker: Worker) {
    this.worker = worker
    this.worker.addEventListener('message', this.handleMessage.bind(this))
    log('constructed; message listener attached')
  }

  private handleMessage(event: MessageEvent<WorkerResponse | WorkerEvent>): void {
    const { request_id, res } = event.data
    const handler = this.listeners.get(request_id)
    log('msg received: request_id=', request_id, 'handler?', !!handler, 'persistent?', this.persistentEvents.has(request_id))
    if (!handler) return
    handler(res)
    if (!this.persistentEvents.has(request_id)) {
      this.listeners.delete(request_id)
    }
  }

  addPersistentListener<E extends EventName>(name: E, handler: EventHandler<E>): void {
    this.persistentEvents.add(name)
    this.listeners.set(name, handler as (data: unknown) => void)
    log('addPersistentListener:', name)
  }

  removePersistentListener(name: EventName): void {
    this.persistentEvents.delete(name)
    this.listeners.delete(name)
  }

  sendMessage<C extends CmdName>(
    cmd:      C,
    data:     CmdDataMap[C],
    transfer: Transferable[] = [],
  ): Promise<CmdResponseMap[C]> {
    const request_id = crypto.randomUUID()
    log('sendMessage:', cmd, 'request_id=', request_id, 'transferables=', transfer.length)
    return new Promise((resolve) => {
      this.listeners.set(request_id, (res) => resolve(res as CmdResponseMap[C]))
      const message: WorkerRequest<C> = { cmd, data, request_id }
      this.worker.postMessage(message, transfer)
    })
  }

  terminate(): void {
    this.worker.terminate()
    this.listeners.clear()
    this.persistentEvents.clear()
  }
}
