#!/usr/bin/env python3
"""Depth bands of game ends for devrun tags, and how many games saw / entered the Quest portal level.

usage: bands.py TAG [TAG ...]
"""
import collections
import json
import os
import sys
from pathlib import Path

# the workspace holding evals/ and devruns/: outside the repo, so run outputs never land in the solution root
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])
BANDS = [(1, 4), (5, 9), (10, 15), (16, 20), (21, 24), (25, 29), (30, 99)]


def main():
    for tag in sys.argv[1:]:
        d = ROOT / "devruns" / tag
        rows = []
        for f in sorted(d.glob("*__*.json")):
            r = json.loads(f.read_text())
            log = d / "logs" / f"bot_{f.stem}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            seen = "portal level" in text.lower() or "telepathic" in text.lower()
            rows.append((r, seen))
        n = len(rows)
        if not n:
            continue
        print(f"== {tag}: {n} games, mean {sum(r['progress'] for r, _ in rows) / n:.4f}")
        band = collections.Counter()
        seen_band = collections.Counter()
        for r, seen in rows:
            for lo, hi in BANDS:
                if lo <= r["depth"] <= hi:
                    band[(lo, hi)] += 1
                    seen_band[(lo, hi)] += seen
        for lo, hi in BANDS:
            print(f"  D{lo}-{hi}: {band[(lo, hi)]:4d} ({band[(lo, hi)] / n:.2f})  portal level seen: {seen_band[(lo, hi)]}")
        homes = sum(1 for r, _ in rows if (r.get("milestone") or "").startswith("Home"))
        print(f"  Home milestone reached: {homes}")


if __name__ == "__main__":
    main()
