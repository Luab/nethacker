#!/bin/sh
# usage: [JF_CFG=..] [JF_ROLE_CFG=..] [NHBOT_ROUTE=..] [PAIRS=file] tools/ab2.sh TAG IDS SEEDS JOBS  -- devrun of the current solution
TAG=$1; IDS=$2; SEEDS=$3; J=$4
W=${NH_WORK:-/Users/semyon/Nethack}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$W"
export JF_CFG JF_ROLE_CFG NHBOT_ROUTE
exec dev/.venv/bin/python "$REPO/tools/devrun.py" "$REPO" "$TAG" --ids "$IDS" --seeds "$SEEDS" -j "$J" ${PAIRS:+--pairs "$PAIRS"} >> "devruns/$TAG.log" 2>&1
