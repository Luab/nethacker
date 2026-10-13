#!/usr/bin/env python3
"""Register per-identity evidence (from `nethackers eval`, as written by sweep.py) with the hub.

Unlike `nethackers register`, this sends the repo's nethackers.solution.json as the manifest, so the
parents/influences attribution reaches the hub. Run it with the nethackers tool's interpreter:

    ~/.local/share/uv/tools/nethackers/bin/python tools/register.py TAG COMMIT [identity ...]

TAG names evals/TAG/*.json in the workspace; COMMIT must be the pushed commit those evals scored. The manifest is
read from $MANIFEST (default: this checkout's nethackers.solution.json) -- use the evaluated commit's own.
Already-registered identities are listed in evals/TAG.registered and skipped.
"""
import json
import os
import sys
from pathlib import Path

from nethackers import cli
from nethackers.hubclient.client import HubClient

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])
HUB = os.environ.get("NETHACKERS_HUB", "https://nethackers.dunnolab.ai")


def main():
    tag, commit, *only = sys.argv[1:]
    token = cli._authed_token()
    if token is None:
        sys.exit("not logged in: run `nethackers login`")
    manifest = json.loads(Path(os.environ.get("MANIFEST") or REPO / "nethackers.solution.json").read_text())
    manifest = {"root": ".", "entrypoint": manifest.get("entrypoint", "bot.py"),
                "parents": manifest.get("parents", []), "influences": manifest.get("influences", [])}
    reference = {"repo": "github.com/zarutskiysy/nethack", "commit": commit}
    client = HubClient(HUB)
    files = sorted((ROOT / "evals" / tag).glob("*.json"))
    done_path = ROOT / "evals" / f"{tag}.registered"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    for f in files:
        if (only and f.stem not in only) or f.stem in done:
            continue
        evidence = json.loads(f.read_text())
        bad = [r for r in evidence["results"] if r["status"] != "completed"]
        if bad:
            print(f"{f.stem}: SKIPPED, {len(bad)} episodes not completed", flush=True)
            continue
        try:
            res = client.register(token=token, reference=reference, manifest=manifest, evidence=evidence)
            print(f"{f.stem}: {evidence['mean_progress']:.4f} registered {res}", flush=True)
            with open(done_path, "a") as fh:
                fh.write(f.stem + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"{f.stem}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
