#!/usr/bin/env python3
"""Summarize evals/TAG/*.json: per-identity mean, death causes, depth histogram.

usage: summary.py TAG [TAG2]   (with TAG2: per-identity comparison)
"""
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

# the workspace holding evals/ and devruns/: outside the repo, so run outputs never land in the solution root
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])


def load(tag):
    out = {}
    for f in sorted((ROOT / "evals" / tag).glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def main():
    a = load(sys.argv[1])
    if len(sys.argv) > 2:
        b = load(sys.argv[2])
        common = sorted(set(a) & set(b))
        for k in common:
            print(f"{k:18s} {a[k]['mean_progress']:.4f} {b[k]['mean_progress']:.4f} {b[k]['mean_progress'] - a[k]['mean_progress']:+.4f}")
        if common:
            ma = statistics.fmean(a[k]['mean_progress'] for k in common)
            mb = statistics.fmean(b[k]['mean_progress'] for k in common)
            print(f"{'MEAN':18s} {ma:.4f} {mb:.4f} {mb - ma:+.4f}  n={len(common)}")
        return
    causes, depths, statuses = Counter(), Counter(), Counter()
    for k, d in a.items():
        rs = d["results"]
        print(f"{k:18s} {d['mean_progress']:.4f}  depths={sorted(r['max_depth'] for r in rs)}")
        for r in rs:
            causes[(r.get("cause_of_death") or r["status"]).split(",")[0]] += 1
            depths[r["max_depth"]] += 1
            statuses[r["status"]] += 1
    if a:
        print("MEAN", statistics.fmean(d["mean_progress"] for d in a.values()), "n=", len(a))
    print(statuses)
    print("depth histogram:", sorted(depths.items()))
    for c, n in causes.most_common(40):
        print(f"{n:4d} {c}")


if __name__ == "__main__":
    main()
