#!/bin/sh
# usage: tools/ab.sh TAG IDS SEEDS JOBS [JF_CFG json]  -- devrun of the current solution with an optional config override
TAG=$1; IDS=$2; SEEDS=$3; J=$4; CFG=$5
W=${NH_WORK:-/Users/semyon/Nethack}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"
JF_CFG="$CFG" exec dev/.venv/bin/python "$REPO/tools/devrun.py" "$REPO" "$TAG" --ids "$IDS" --seeds "$SEEDS" -j "$J" >> "devruns/$TAG.log" 2>&1
