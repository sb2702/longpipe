// Audio passthrough — main thread only. Worker never sees audio.
//
// 'passthrough': output stream carries the input's audio tracks unchanged.
// 'drop':        output stream is video-only.
//
// Future: react to mid-call audio track replacement (e.g., user changes
// mic). v0.1 captures the audio track set at init and doesn't update.

export type AudioMode = 'passthrough' | 'drop'

export function buildOutputStream(
  videoTrack:  MediaStreamTrack,
  inputStream: MediaStream,
  mode:        AudioMode,
): MediaStream {
  const tracks: MediaStreamTrack[] = [videoTrack]
  if (mode === 'passthrough') tracks.push(...inputStream.getAudioTracks())
  return new MediaStream(tracks)
}
