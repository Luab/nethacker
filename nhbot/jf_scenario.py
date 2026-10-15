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
  form              the polymorph form a setup's wizard #polyself gave us (e.g. "xorn")
  hp_before_poly    [hp, hpmax] of our own form under it
  self_mon          the monster our own form shows as (e.g. "valkyrie"), when a setup polymorphed us first
  poly_control      the setup showed polymorph control (the rings worn now give it)
  identity          {"role": "VALKYRIE", "race": "DWARF", "gender": "FEMALE", "alignment": "LAWFUL"} when the
                    attribute parse can't read them (a polymorphed start)
  medusa_level      [dnum, dlvl] of Medusa's level
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
        if STATE.get('form'):
            # a setup that polymorphs us (wizard #polyself) ran before the bot saw 'You turn into a ...!', which is
            # how castle_cross.note_message learns the form (the glyph test takes the form's glyph for our own)
            agent._cfp_form = STATE['form']
        if STATE.get('identity'):
            # character.parse fails on a polymorphed hero's attributes ('You are actually a ...'), leaving role,
            # race, gender and alignment unknown (KeyError: None in the cannibalism check, vxw1)
            from .character import Character
            ch = agent.character
            for field, value in STATE['identity'].items():
                if getattr(ch, field, None) is None:
                    setattr(ch, field, getattr(Character, value))
        if STATE.get('self_mon'):
            # ...and character.parse read the form's glyph as our own (the setup polymorphed us first), so
            # prop.polymorph stayed true after the form ended and the dwarf waited it out for good (vxx3 s1)
            from .glyph import MON
            agent.character.self_glyph = MON.from_name(STATE['self_mon'])
        if STATE.get('hp_before_poly'):
            # ...and character.update took the HP it last saw in our own form, before the bot's first step: none
            agent.character.hp_before_poly = tuple(STATE['hp_before_poly'])
        if STATE.get('poly_control'):
            # the setup's wizard #polyself asked 'Become what kind of monster?' before the bot ran
            agent._note_poly_control()
        if STATE.get('medusa_level'):
            dive.medusa_level = tuple(STATE['medusa_level'])
        if STATE.get('castle_known'):
            # a setup that polymorphs us on the castle into a form that can't dig (lift-ready's flyer tests): the
            # castle is recognised by a dig that form can't make -- castle_logic.note_level marks it instead
            dive._scenario_castle = True
        agent.log(f'SCENARIO start: {STATE} -> diving={dive.diving} mines_done={dive.mines_done} '
                  f'milestone={gl.milestone.name}')
    except Exception as e:   # a typo in a dev scenario must not kill the agent thread
        agent.log(f'SCENARIO state not applied ({type(e).__name__}: {e}): {STATE}')
