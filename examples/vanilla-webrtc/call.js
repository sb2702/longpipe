// A loopback "call": two RTCPeerConnections in the same page, pc1 → pc2, with
// the signaling done by passing objects directly. In a real app pc2 is on the
// other side of your signaling server; everything else is the same.

export async function startLoopbackCall(localStream, { onRemoteTrack } = {}) {
  const pc1 = new RTCPeerConnection()
  const pc2 = new RTCPeerConnection()
  // A peer can't take ICE candidates before it has a remote description, so
  // queue them until then (a real app's signaling has the same rule).
  const relay = (to) => {
    const queue = []
    const add = (c) => to.addIceCandidate(c).catch((err) => console.warn('addIceCandidate:', err))
    return {
      onCandidate: (e) => { if (e.candidate) (to.remoteDescription ? add(e.candidate) : queue.push(e.candidate)) },
      flush: () => { for (const c of queue.splice(0)) add(c) },
    }
  }
  const toPc2 = relay(pc2), toPc1 = relay(pc1)
  pc1.onicecandidate = toPc2.onCandidate
  pc2.onicecandidate = toPc1.onCandidate

  const remoteStream = new MediaStream()
  pc2.ontrack = (e) => {
    remoteStream.addTrack(e.track)
    onRemoteTrack?.(e.track, remoteStream)
  }

  // One sender per kind. Keep them: replaceTrack() is how you swap a track
  // later (camera switch, or Longpipe's onTrackChanged) without renegotiating.
  const senders = {}
  for (const track of localStream.getTracks()) senders[track.kind] = pc1.addTrack(track, localStream)

  const offer = await pc1.createOffer()
  await pc1.setLocalDescription(offer)
  await pc2.setRemoteDescription(offer)
  toPc2.flush()
  const answer = await pc2.createAnswer()
  await pc2.setLocalDescription(answer)
  await pc1.setRemoteDescription(answer)
  toPc1.flush()

  return {
    pc1, pc2, senders, remoteStream,
    close() { pc1.close(); pc2.close() },
  }
}

// The few RTP numbers worth watching: what pc1 sent and what pc2 decoded, plus
// the cumulative timing counters behind the latency breakdown (see latencyBreakdown).
export async function callStats(call) {
  const out = { video: {}, audio: {} }
  const [sent, recv] = await Promise.all([call.pc1.getStats(), call.pc2.getStats()])
  sent.forEach((r) => {
    if (r.type === 'outbound-rtp' && r.kind === 'video') {
      out.video.framesSent = r.framesSent ?? 0
      out.video.framesEncoded = r.framesEncoded ?? 0
      out.video.totalEncodeTime = r.totalEncodeTime ?? 0          // seconds, summed over framesEncoded
    }
    if (r.type === 'outbound-rtp' && r.kind === 'audio') out.audio.packetsSent = r.packetsSent ?? 0
  })
  recv.forEach((r) => {
    if (r.type === 'inbound-rtp' && r.kind === 'video') {
      out.video.framesDecoded = r.framesDecoded ?? 0
      out.video.framesReceived = r.framesReceived ?? 0
      out.video.frameWidth = r.frameWidth ?? 0
      out.video.frameHeight = r.frameHeight ?? 0
      out.video.totalDecodeTime = r.totalDecodeTime ?? 0          // seconds, summed over framesDecoded
      out.video.jitterBufferDelay = r.jitterBufferDelay ?? 0      // seconds, summed over jitterBufferEmittedCount
      out.video.jitterBufferEmittedCount = r.jitterBufferEmittedCount ?? 0
    }
    if (r.type === 'inbound-rtp' && r.kind === 'audio') {
      out.audio.packetsReceived = r.packetsReceived ?? 0
      out.audio.totalSamplesReceived = r.totalSamplesReceived ?? 0
      out.audio.audioLevel = r.audioLevel ?? 0
      out.audio.totalAudioEnergy = r.totalAudioEnergy ?? 0
    }
  })
  return out
}

// Per-frame encode / jitter-buffer / decode time in ms over the interval
// between two callStats() samples (deltas of the cumulative counters, so a
// model switch or bypass shows up within a second instead of being averaged
// into the call's lifetime). null where the interval had no frames.
export function latencyBreakdown(prev, cur) {
  const per = (sumKey, countKey) => {
    const n = (cur.video[countKey] ?? 0) - (prev?.video[countKey] ?? 0)
    const t = (cur.video[sumKey] ?? 0) - (prev?.video[sumKey] ?? 0)
    return n > 0 && t >= 0 ? (t / n) * 1000 : null
  }
  return {
    encodeMs: per('totalEncodeTime', 'framesEncoded'),
    jitterMs: per('jitterBufferDelay', 'jitterBufferEmittedCount'),
    decodeMs: per('totalDecodeTime', 'framesDecoded'),
  }
}
