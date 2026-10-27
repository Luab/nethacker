#!/bin/sh
# Run tools/devrun.py inside the pinned amd64 arena image (same dungeons as the hub's scoring).
# usage: tools/dockrun.sh <devrun args...>   e.g. tools/dockrun.sh . pub_arc --ids arc-hum-law-mal --seeds 0-14 --secret public --eval-id local -j 4
W=${NH_WORK:-/Users/semyon/Nethack}
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMG=ghcr.io/dunnolab/nethackers-arena@sha256:0d0b0e779ebda4a05b5a22b39cfafcd2d2d9ef739ea60d9aae79052d55c767ae
exec docker run --rm --platform linux/amd64 -v "$W:$W" -w "$REPO" -e NH_WORK="$W" \
  -e JF_CFG="$JF_CFG" --entrypoint /opt/nethackers/.venv/bin/python "$IMG" tools/devrun.py "$@"
