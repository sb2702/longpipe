#!/usr/bin/env bash
# Run Playwright's WebKit on an Ubuntu 22.04 machine that's missing libavif13
# (Playwright says "Host system is missing dependencies") when you can't sudo.
# Downloads the .debs with `apt-get download` (no root), unpacks them into
# tests/e2e/.cache/webkit-libs, and prints an LD_PRELOAD to export. The
# MiniBrowser wrapper overwrites LD_LIBRARY_PATH, so preloading is the way in.
#
#   eval "$(tests/e2e/tools/webkit-libs-nosudo.sh)"
#   npm run test:e2e -- --project=webkit
#
# With sudo, prefer: sudo npx playwright install-deps webkit. CI does that.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dir="$here/.cache/webkit-libs"
mkdir -p "$dir"
cd "$dir"
if [ ! -f .done ]; then
  apt-get download libavif13 libgav1-0 libyuv0 libabsl20210324 >&2
  for d in *.deb; do dpkg -x "$d" root; done
  touch .done
fi
L="$dir/root/usr/lib/x86_64-linux-gnu"
pre=$(LD_LIBRARY_PATH="$L" ldd "$L/libavif.so.13" | awk '{print $3}' | grep "^$L" | tac | tr '\n' ':')
echo "export LD_PRELOAD=${pre}$L/libavif.so.13"
echo "export LD_LIBRARY_PATH=$L\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
echo "export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1"
