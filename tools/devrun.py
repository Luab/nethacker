#!/usr/bin/env python3
"""Native dev runner: play a bot on held-out seeds and keep death diagnostics.

usage: dev/.venv/bin/python tools/devrun.py BOT_DIR OUT_TAG --ids val-dwa-law-fem,wiz-hum-neu-mal
           [--seeds 0-9] [--eval-id dev1] [-j 8]
Writes devruns/OUT_TAG/<ident>__<seed>.json with progress, depth, cause, last messages and
the final screen. Seeds come from nethackers' HMAC derivation with secret 'dev' and the given
evaluation id, so they never overlap the published batches.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

# the workspace holding evals/ and devruns/: outside the repo, so run outputs never land in the solution root
ROOT = Path(os.environ.get("NH_WORK") or Path(__file__).resolve().parents[2])


def play(args):
    bot_dir, ident, seed, eval_id, out_path, max_steps, secret = args
    os.environ["JF_LOG_DIR"] = str(Path(out_path).parent / "logs")
    os.environ["JF_EPISODE"] = f"{ident}__{seed}"
    import random
    import warnings
    warnings.filterwarnings("ignore")
    from nethackers.arena.environment import make_environment
    from nethackers.arena.seeds import trajectory_spec
    spec = trajectory_spec(secret, eval_id, seed)
    sys.path.insert(0, bot_dir)
    os.chdir(bot_dir)
    random.seed(spec.bot_seed)
    import numpy as np
    np.random.seed(spec.bot_seed % (1 << 32))
    import bot
    env = make_environment(max_steps, 10_000, ident)
    agent = bot.make_agent()
    t0 = time.time()
    obs = env.reset(spec)
    agent.reset(obs)
    msgs, steps, error, last = [], 0, None, obs
    live_bl, live_screen = None, None   # the last observation before the end screens (blstats zeroed there)
    try:
        while True:
            action = agent.act(obs)
            obs, _r, term, trunc = env.step(action)
            last = obs
            steps += 1
            if obs["blstats"][10] > 0 and obs["blstats"][20] > 0:
                live_bl = obs["blstats"].copy()
                if steps % 50 == 0 or obs["blstats"][10] < obs["blstats"][11] // 3:
                    live_screen = obs["tty_chars"].copy()
            m = bytes(obs["message"]).split(b"\0")[0].decode("latin-1").strip()
            if m and (not msgs or msgs[-1][1] != m):
                msgs.append((int(obs["blstats"][20]), m))
                msgs = msgs[-int(os.environ.get("DEVRUN_KEEP", "60")):]
            if term or trunc:
                break
    except Exception:
        error = traceback.format_exc()[-3000:]
    m = env.metrics()
    screen = bytes(last["tty_chars"]).decode("latin-1")
    screen = "\n".join(screen[i:i + 80].rstrip() for i in range(0, len(screen), 80))
    rec = dict(ident=ident, seed=seed, progress=m.progress, milestone=m.milestone, depth=m.max_depth,
               turns=m.turns, steps=steps, cause=m.cause_of_death, end_status=m.end_status,
               wall=time.time() - t0, error=error, messages=msgs, screen=screen,
               blstats=[int(x) for x in last["blstats"]],
               live_bl=[int(x) for x in live_bl] if live_bl is not None else None,
               live_screen=(lambda t: "\n".join(t[i:i + 80].rstrip() for i in range(0, len(t), 80)))(
                   bytes(live_screen).decode("latin-1")) if live_screen is not None else None)
    Path(out_path).write_text(json.dumps(rec))
    try:
        agent.close()
    except Exception:
        pass
    env.close()
    return f"{ident} s{seed}: {m.progress:.3f} D{m.max_depth} T{m.turns} {m.cause_of_death} ({time.time() - t0:.0f}s)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bot")
    p.add_argument("tag")
    p.add_argument("--ids", required=True)
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--eval-id", default="dev1")
    p.add_argument("--secret", default="dev", help="'public' with --eval-id local replays the published batches")
    p.add_argument("-j", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--pairs", help="file of 'ident__seed' lines: play only these games (--ids still required)")
    a = p.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    ids = a.ids.split(",")
    if ids == ["all"]:
        ids = [o["name"] for o in json.loads((ROOT / "evals" / "objectives.json").read_text())
               if o["kind"] == "identity"]
    out = ROOT / "devruns" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    # play a frozen copy: edits to the bot while the run goes on must not leak into it
    snap = out / "bot_snapshot"
    if not snap.exists():
        import shutil
        shutil.copytree(Path(a.bot).resolve(), snap, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    (out / "jf_cfg.txt").write_text(os.environ.get("JF_CFG", "") + "\n" + os.environ.get("JF_ROLE_CFG", ""))
    bot_dir = str(snap)
    pairs = [(i, s) for s in seeds for i in ids]
    if a.pairs:   # only these games: lines 'ident__seed'
        pairs = [(i, int(s)) for i, s in (ln.strip().split("__") for ln in open(a.pairs) if ln.strip())]
    jobs = [(bot_dir, i, s, a.eval_id, str(out / f"{i}__{s}.json"), a.max_steps, a.secret)
            for i, s in pairs if not (out / f"{i}__{s}.json").exists()]
    # str hashes are salted per process unless PYTHONHASHSEED is set, and the bot's choices depend on set iteration
    # order in places: without this two runs of one game diverge (v2a/rep1/rep2 arc-dwa-law-fem s204: D29 vs D24),
    # and so do the two arms of an A/B long before the change under test acts
    os.environ.setdefault("PYTHONHASHSEED", "0")
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.j, maxtasksperchild=1) as pool:
        for line in pool.imap_unordered(play, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
