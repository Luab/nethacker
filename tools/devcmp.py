#!/usr/bin/env python3
"""Compare devrun tags episode by episode (paired on ident+seed).

usage: devcmp.py TAG_A [TAG_B]   -- one tag: per-identity summary and death causes; TAG may be a+b (union)
"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# the workspace holding evals/ and devruns/: outside the repo, so run outputs never land in the solution root
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])


def load(tag):
    """One tag, or several joined with '+' (their union)."""
    out = {}
    for t in tag.split("+"):
        for f in (ROOT / "devruns" / t).glob("*__*.json"):
            r = json.loads(f.read_text())
            out[(r["ident"], r["seed"])] = r
    return out


def main():
    a = load(sys.argv[1])
    if len(sys.argv) == 2:
        by = defaultdict(list)
        for r in a.values():
            by[r["ident"]].append(r)
        for k in sorted(by):
            rs = by[k]
            print(f"{k:18s} n={len(rs):3d} mean={statistics.fmean(r['progress'] for r in rs):.4f} "
                  f"depths={sorted(r['depth'] for r in rs)}")
        print(f"ALL n={len(a)} mean={statistics.fmean(r['progress'] for r in a.values()):.4f}")
        print("errors:", sum(1 for r in a.values() if r["error"]))
        c = Counter((r["cause"] or r["end_status"] or "?") for r in a.values())
        for k, n in c.most_common(30):
            print(f"{n:4d} {k}")
        return
    b = load(sys.argv[2])
    keys = sorted(set(a) & set(b))
    if not keys:
        print("no common episodes")
        return
    diffs = [b[k]["progress"] - a[k]["progress"] for k in keys]
    per = defaultdict(list)
    for k, d in zip(keys, diffs):
        per[k[0]].append(d)
    for ident in sorted(per):
        ds = per[ident]
        print(f"{ident:18s} n={len(ds):3d} delta={statistics.fmean(ds):+.4f}")
    ma = statistics.fmean(a[k]["progress"] for k in keys)
    mb = statistics.fmean(b[k]["progress"] for k in keys)
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    print(f"n={len(keys)} A={ma:.4f} B={mb:.4f} delta={mb - ma:+.4f} (se {se:.4f}) wins={wins} losses={losses}")


if __name__ == "__main__":
    main()
