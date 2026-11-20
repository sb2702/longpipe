#!/usr/bin/env bash
# Downloads the official livekit-server release binary into examples/livekit/.server/
# (gitignored). No sudo, no system install. Linux amd64/arm64 only: LiveKit ships no
# macOS binary (on a Mac: `brew install livekit`, or tunnel to a Linux server, see README).
set -euo pipefail
VERSION="${LIVEKIT_VERSION:-1.13.11}"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ARCH=linux_amd64 ;;
  Linux-aarch64) ARCH=linux_arm64 ;;
  *) echo "no livekit-server binary for $(uname -s)-$(uname -m); see README" >&2; exit 1 ;;
esac
DIR="$(cd "$(dirname "$0")" && pwd)/.server"
FILE="livekit_${VERSION}_${ARCH}.tar.gz"
URL="https://github.com/livekit/livekit/releases/download/v${VERSION}"
mkdir -p "$DIR" && cd "$DIR"
[ -x livekit-server ] && ./livekit-server --version | grep -q "$VERSION" && { echo "$DIR/livekit-server"; exit 0; }
curl -fsSL -o "$FILE" "$URL/$FILE"
curl -fsSL "$URL/checksums.txt" | grep " $FILE\$" | sha256sum -c - >&2
tar -xzf "$FILE" livekit-server && rm "$FILE"
echo "$DIR/livekit-server"
