"""Opt-in diagnostics for dev runs; a no-op in the arena.

Enabled only when JF_LOG_DIR is set (dev/eval.sh sets it; the official sandbox
does not). Each episode appends to <JF_LOG_DIR>/bot_<JF_EPISODE>.log.
"""
import os

_DIR = os.environ.get("JF_LOG_DIR")
_fh = None


def enabled():
    return _DIR is not None


def log(msg):
    global _fh
    if _DIR is None:
        return
    try:
        if _fh is None:
            os.makedirs(_DIR, exist_ok=True)
            episode = os.environ.get("JF_EPISODE", str(os.getpid()))
            _fh = open(os.path.join(_DIR, f"bot_{episode}.log"), "a", buffering=1)
        _fh.write(msg + "\n")
    except OSError:
        pass
