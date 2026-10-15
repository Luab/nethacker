#!/usr/bin/env python3
"""Evaluate a bot directory on many identities with `nethackers eval`.

usage: sweep.py BOT_DIR TAG [identity ...]   (no identities: all 73)
Writes evals/TAG/<identity>.json; skips identities already evaluated.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# the workspace holding evals/ and devruns/: outside the repo, so run outputs never land in the solution root
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])
HUB = "https://nethackers.dunnolab.ai"


def identities():
    cache = ROOT / "evals" / "objectives.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(f"{HUB}/objectives").read())
    return [o["name"] for o in json.loads(cache.read_text()) if o["kind"] == "identity"]


def main():
    bot, tag, *ids = sys.argv[1:]
    out = ROOT / "evals" / tag
    out.mkdir(parents=True, exist_ok=True)
    for ident in ids or identities():
        dest = out / f"{ident}.json"
        if dest.exists():
            continue
        proc = subprocess.run(
            ["nethackers", "--no-tui", "-o", "json", "eval", bot, "--objective", ident],
            capture_output=True, text=True)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(f"{ident}: FAILED rc={proc.returncode} {proc.stderr[-500:]}", flush=True)
            continue
        dest.write_text(proc.stdout)
        errs = sum(r["status"] != "completed" for r in data["results"])
        print(f"{ident}: {data['mean_progress']:.4f} errors={errs}", flush=True)


if __name__ == "__main__":
    main()
