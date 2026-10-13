"""PREP_LOG: checkpoint state lines for the READINESS metric (logging only; it never changes a game).

READINESS (dev/readiness.py; ledger F088) scores how well placed a game is to get past the Castle
(Dlvl 25-29, the bottom of the Dungeons of Doom): alive x the state that decides a pass -- the kit (lifts,
teleport control, wishes, escapes), strength (XL, HP, AC, MC, Excalibur) and food. This module writes that
state into the bot log at fixed checkpoints so the metric can be computed from any dev run:

  * turn checkpoints: the first observation at or after every multiple of TURN_STEP game turns (T2500,
    T5000, ... -- the metric reads T10000/T20000/T30000);
  * depth checkpoints: the first observation on each new deepest level of a dungeon branch (dnum 0 = the
    Dungeons of Doom, 1 = Gehennom, 2 = the Mines, ...), so Dlvl 10, Dlvl 20, the Castle and the Valley
    arrivals are all there.

A line is 'PREP {json}' (keys below, in `_state`). It is written through agent.log, which is a no-op unless
JF_LOG_DIR is set (dev runs only; the arena never sets it), and only when jf_config.PREP_LOG is on.

Game identity (the reason this file exists apart from the strategy code): note() only READS the agent -- the
last observation, blstats and the bot's item knowledge -- and never steps the environment, never calls a
strategy, never touches an RNG and never writes agent state other than its own `_prep_log` bookkeeping
attribute, which nothing else reads. Every failure is caught (the first 3 are logged as 'PREP error').
Pinned-seed checks (dev/prep_idcheck.py): runs with PREP_LOG on replay the flag-off games byte-identically
(msgs_N.txt), e.g. s23 and cand-g on the fresh sets jf43-46, 120/120 games.
"""

import json

import nle.nethack as nh

from . import jf_config, jf_log
from . import objects as O

TURN_STEP = 2500

# Items that decide (or help) a pass, by mechanism. The metric aggregates them into features; see
# `python3 dev/readiness.py --explain` for the weight of each and its evidence. (name, object class or None for a
# unique name)
_TARGETS = {
    # the wish route (tele_route.py): a wand of wishing -> ring of teleport control + cursed teleport scrolls
    'wish_wand': ('wishing', nh.WAND_CLASS),
    'magic_lamp': ('magic lamp', None),
    'charging': ('charging', nh.SCROLL_CLASS),
    # the teleport-control route: TC + a level teleport (cursed/confused scroll of teleportation)
    'tc_ring': ('teleport control', nh.RING_CLASS),
    'tele_scroll': ('teleportation', nh.SCROLL_CLASS),
    'confusion_pot': ('confusion', nh.POTION_CLASS),
    'booze_pot': ('booze', nh.POTION_CLASS),
    # castle moat lifts (power.passage_plan)
    'lev_ring': ('levitation', nh.RING_CLASS),
    'lev_boots': ('levitation boots', None),
    'ww_boots': ('water walking boots', None),
    'lev_pot': ('levitation', nh.POTION_CLASS),
    'mb_amulet': ('amulet of magical breathing', None),
    'cold_wand': ('cold', nh.WAND_CLASS),
    'frost_horn': ('frost horn', None),
    # polymorph (POLY_XORN / CASTLE_POLY): a xorn walks the castle walls and the Valley's stone
    'poly_wand': ('polymorph', nh.WAND_CLASS),
    'poly_pot': ('polymorph', nh.POTION_CLASS),
    'polyctl_ring': ('polymorph control', nh.RING_CLASS),
    'poly_ring': ('polymorph', nh.RING_CLASS),
    # escapes and minotaur stoppers (maze levels, castle landing)
    'scare': ('scare monster', nh.SCROLL_CLASS),
    'dig_wand': ('digging', nh.WAND_CLASS),
    'sleep_wand': ('sleep', nh.WAND_CLASS),
    'tele_wand': ('teleportation', nh.WAND_CLASS),
    'striking_wand': ('striking', nh.WAND_CLASS),
    'opening_wand': ('opening', nh.WAND_CLASS),
    'fire_wand': ('fire', nh.WAND_CLASS),
    'lightning_wand': ('lightning', nh.WAND_CLASS),
    'death_wand': ('death', nh.WAND_CLASS),
    'teleportitis_ring': ('teleportation', nh.RING_CLASS),
    # defence / strength
    'ls_amulet': ('amulet of life saving', None),
    'refl_amulet': ('amulet of reflection', None),
    'refl_shield': ('shield of reflection', None),
    'mr_cloak': ('cloak of magic resistance', None),
    'prot_cloak': ('cloak of protection', None),
    'gdsm': ('gray dragon scale mail', None),
    'sdsm': ('silver dragon scale mail', None),
    'speed_boots': ('speed boots', None),
    'free_action': ('free action', nh.RING_CLASS),
    'conflict_ring': ('conflict', nh.RING_CLASS),
    'regen_ring': ('regeneration', nh.RING_CLASS),
    'slowdig_ring': ('slow digestion', nh.RING_CLASS),
    'unicorn_horn': ('unicorn horn', None),
    'ench_armor': ('enchant armor', nh.SCROLL_CLASS),
    'ench_weapon': ('enchant weapon', nh.SCROLL_CLASS),
    'gain_level': ('gain level', nh.POTION_CLASS),
    'full_healing': ('full healing', nh.POTION_CLASS),
    'genocide': ('genocide', nh.SCROLL_CLASS),
    'identify': ('identify', nh.SCROLL_CLASS),
    'remove_curse': ('remove curse', nh.SCROLL_CLASS),
    'magic_marker': ('magic marker', None),
}
_OBJS = None   # target key -> object, resolved on first use (a name missing from this build is skipped)

# Groups scored per ITEM (one unknown pair of boots may be levitation or water walking: P = the sum), then
# 'g_<group>': [n identified, P(at least one unidentified item is in the group)].
_GROUPS = {
    'lift': ('lev_ring', 'lev_boots', 'ww_boots', 'lev_pot', 'mb_amulet', 'cold_wand', 'frost_horn'),
    'lastlift': ('lev_ring', 'lev_boots', 'ww_boots', 'mb_amulet'),     # stays on for the whole crossing
    'polysrc': ('poly_wand', 'poly_pot', 'poly_ring'),
    'minostop': ('scare', 'sleep_wand', 'tele_wand', 'dig_wand', 'tele_scroll'),  # minotaurs ignore Elbereth
}


def _objs():
    global _OBJS
    if _OBJS is None:
        out = {}
        for key, (name, cls) in _TARGETS.items():
            try:
                out[key] = O.from_name(name, cls) if cls is not None else O.from_name(name)
            except Exception:
                pass
        _OBJS = out
    return _OBJS


def _weight(obj):
    """Generation weight of an object (rings and some tools carry none: equally likely), as power.p_of."""
    p = getattr(obj, 'prob', None)
    return p if p else 1


def _p_is(item, obj):
    """P(item is `obj`) by generation weight among the identities the bot's knowledge still allows (price
    ranges, engrave tests, discoveries): 1.0 for an identified item."""
    objs = item.objs
    if obj not in objs:
        return 0.0
    if len(objs) == 1:
        return 1.0
    total = sum(_weight(o) for o in objs)
    return _weight(obj) / total if total > 0 else 0.0


def _all_items(agent):
    """Inventory items plus the contents of checked containers: [(letter or '<letter>/', item)]."""
    inv = agent.inventory.items
    out = []
    for letter, item in zip(inv.all_letters, inv.all_items):
        out.append((letter, item))
        content = getattr(item, 'content', None)
        if content is not None and not callable(content):
            for sub in getattr(content, 'items', ()) or ():
                out.append((letter + '/', sub))
    return out


def _knowledge(agent):
    """Per target: [n identified (stack counts summed), P(at least one unidentified item is it)], only for
    targets present; and the bot's own identity for items whose text doesn't name them (price-identified,
    engrave-tested): letter -> name, or the candidate list when there are at most 4."""
    objs = _objs()
    kn = {}
    ids = {}
    for letter, item in _all_items(agent):
        if item.category == nh.COIN_CLASS:
            continue
        count = max(1, int(getattr(item, 'count', 1) or 1))
        p_item = {}
        for key, obj in objs.items():
            p = _p_is(item, obj)
            if p <= 0:
                continue
            p_item[key] = p
            n_known, p_none = kn.get(key, (0, 1.0))
            if p >= 1.0:
                n_known += count
            else:
                p_none *= 1.0 - p
            kn[key] = (n_known, p_none)
        for group, keys in _GROUPS.items():
            p = min(1.0, sum(p_item.get(k, 0.0) for k in keys))
            if p <= 0:
                continue
            n_known, p_none = kn.get('g_' + group, (0, 1.0))
            if p >= 1.0:
                n_known += count
            else:
                p_none *= 1.0 - p
            kn['g_' + group] = (n_known, p_none)
        text = item.text or ''
        if item.is_unambiguous():
            name = item.object.name
            if name and name not in text:
                ids[letter] = name
        elif len(item.objs) <= 4:
            ids[letter] = sorted(o.name for o in item.objs if o.name)
    return {k: [n, round(1.0 - p_none, 4)] for k, (n, p_none) in kn.items()}, ids


def _inventory_text(agent):
    """The inventory exactly as the game shows it (identities as the hero knows them): ['a - ...', ...]."""
    obs = agent.last_observation
    out = []
    for name, letter in zip(obs['inv_strs'], obs['inv_letters']):
        text = bytes(name).decode(errors='replace').strip('\0')
        if text:
            out.append(f'{chr(letter)} - {text}')
    return sorted(set(out))


def _state(agent, checkpoints):
    bl = agent.blstats
    st = {
        'ck': checkpoints,                       # which checkpoints this observation completes
        't': int(bl.time), 's': int(agent.step_count),
        'dn': int(bl.dungeon_number), 'dl': int(bl.level_number), 'd': int(bl.depth),
        'xl': int(bl.experience_level), 'xp': int(bl.experience_points),
        'hp': int(bl.hitpoints), 'hpm': int(bl.max_hitpoints), 'pw': int(bl.energy), 'pwm': int(bl.max_energy),
        'ac': int(bl.armor_class), 'st': int(bl.strength), 'st%': int(bl.strength_percentage),
        'dx': int(bl.dexterity), 'co': int(bl.constitution), 'in': int(bl.intelligence), 'wi': int(bl.wisdom),
        'ch': int(bl.charisma), 'au': int(bl.gold), 'hu': int(bl.hunger_state), 'enc': int(bl.carrying_capacity),
        'cond': int(bl.prop_mask), 'al': int(getattr(bl, 'alignment', 0)),
    }
    parts = [('inv', _inventory_text)]
    for key, func in parts:
        try:
            st[key] = func(agent)
        except Exception as e:
            st[key + '_err'] = f'{type(e).__name__}: {str(e)[:80]}'
    try:
        st['kn'], st['bid'] = _knowledge(agent)
    except Exception as e:
        st['kn_err'] = f'{type(e).__name__}: {str(e)[:80]}'
    # carried nutrition (identified food only; corpses count as the bot's parser names them)
    nut = 0
    for _, item in _all_items(agent):
        try:
            if item.is_food() and item.is_unambiguous():
                nut += int(item.object.nutrition) * max(1, int(item.count or 1))
        except Exception:
            pass
    st['nut'] = nut
    # what the bot knows about itself (attributes only; each is optional in older code)
    ch = getattr(agent, 'character', None)
    intr = {}
    for attr in ('telepathic', 'teleportitis', 'teleport_control', 'is_lycanthrope'):
        v = getattr(ch, attr, None)
        if v:
            intr[attr] = True
    # TC_ROUTE's own findings (power_route.RouteState, read directly: power_route.state() would create it):
    # the tengu intrinsic, and ring appearances proven to give teleport control at a TC prompt
    ps = getattr(agent, '_proute', None)
    if ps is not None:
        if getattr(ps, 'tc_intrinsic', False):
            intr['tc_intrinsic'] = True
        tc_glyphs = getattr(ps, 'tc_glyphs', None) or ()
        try:
            if any(item.category == nh.RING_CLASS and item.glyphs[0] in tc_glyphs
                   for _, item in _all_items(agent)):
                intr['tc_ring_proven'] = True
        except Exception:
            pass
    st['intr'] = intr
    st['pray'] = [getattr(agent, 'last_prayer_turn', None), bool(getattr(agent, 'prayer_failed', False))]
    dive = getattr(getattr(agent, 'global_logic', None), 'dive', None)
    st['dv'] = bool(getattr(dive, 'diving', False))
    return st


def _plain(value):
    """json.dumps fallback: numpy scalars (blstats fields, turn counters) -> Python numbers, anything else -> str."""
    item = getattr(value, 'item', None)
    return item() if callable(item) else str(value)


class _Book:
    """Per-game trigger bookkeeping (stored on the agent as `_prep_log`)."""

    def __init__(self):
        self.next_turn = TURN_STEP
        self.max_depth = {}      # dnum -> deepest depth logged
        self.errors = 0


def note(agent):
    """Agent.update calls this on every observation when jf_config.PREP_LOG is on. Cheap unless a
    checkpoint is due; never raises."""
    if not jf_config.PREP_LOG or not jf_log.enabled():
        return
    try:
        book = getattr(agent, '_prep_log', None)
        if book is None:
            book = agent._prep_log = _Book()
        bl = agent.blstats
        checkpoints = []
        if bl.time >= book.next_turn:
            checkpoints.append(f'T{(bl.time // TURN_STEP) * TURN_STEP}')
            book.next_turn = (bl.time // TURN_STEP + 1) * TURN_STEP
        dnum, depth = int(bl.dungeon_number), int(bl.depth)
        if depth > book.max_depth.get(dnum, 0):
            book.max_depth[dnum] = depth
            checkpoints.append(f'D{dnum}:{depth}')
        if not checkpoints:
            return
        agent.log('PREP ' + json.dumps(_state(agent, checkpoints), separators=(',', ':'), sort_keys=True,
                                       default=_plain))
    except Exception as e:
        book = getattr(agent, '_prep_log', None)
        if book is not None and book.errors < 3:
            book.errors += 1
            try:
                agent.log(f'PREP error {type(e).__name__}: {str(e)[:200]}')
            except Exception:
                pass
