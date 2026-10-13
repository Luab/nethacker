"""Dev scenario harness hook (dev/scenario.py); a no-op in the arena.

The harness starts a game mid-way (wizard-mode setup: XL, inventory, level teleport to e.g. the castle,
then wizard mode off) and passes the bot state that game should have as JSON in JF_SCENARIO, e.g.
JF_SCENARIO='{"diving": true}'. The arena never sets JF_SCENARIO, so a submission always has
STATE = None and nothing here changes its behaviour.

Keys:
  diving            (default true) the dive phase is on from the first action
  mines_done        (default true) the dive does not take the Mines route
  milestone         a global_logic.Milestone name (e.g. "GO_DOWN")
  last_prayer_turn  what the bot believes about its last prayer (None: never prayed)
  prayer_failed     the bot believes a prayer has failed (god angry)
"""
import json
import os

_raw = os.environ.get('JF_SCENARIO')
STATE = json.loads(_raw) if _raw else None


def active():
    return STATE is not None


def apply(agent):
    """Set the scenario's bot state; called on every agent start (a driver restart re-applies it)."""
    if STATE is None:
        return
    try:
        from .global_logic import Milestone
        gl = agent.global_logic
        dive = gl.dive
        if STATE.get('diving', True):
            dive.diving = True
            dive.mines_done = bool(STATE.get('mines_done', True))
        if STATE.get('milestone'):
            gl.milestone = Milestone[STATE['milestone']]
        if 'last_prayer_turn' in STATE:
            agent.last_prayer_turn = STATE['last_prayer_turn']
        if 'prayer_failed' in STATE:
            agent.prayer_failed = bool(STATE['prayer_failed'])
        agent.log(f'SCENARIO start: {STATE} -> diving={dive.diving} mines_done={dive.mines_done} '
                  f'milestone={gl.milestone.name}')
    except Exception as e:   # a typo in a dev scenario must not kill the agent thread
        agent.log(f'SCENARIO state not applied ({type(e).__name__}: {e}): {STATE}')
