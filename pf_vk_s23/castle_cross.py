"""castle-first-pass: getting a lifted hero off the castle's west landing and into the moat before it dies there.

Why (castle-first-pass, 2026-09-27; NetHack 3.6.6 castle.des / sp_lev.c / trap.c / dig.c):

* Real Dlvl-29 castle arrivals die on the west side within 5-50 turns, before castle_logic's plan (which runs at the
  bottom of the preempt chain, under fight2 / elbereth_rest / the scare hold) ever tries an item. cfp-p2-jf16 s0
  (pinned clock): landed T10981 at map (-2,2) with an uncursed ring of levitation among 3 unknown rings, a master
  lich (a covetous court monster: it teleports next to the hero, wizard.c tactics -> mnexto) arrived at T10985, a
  soldier ant and an elf-lord joined; dead at T11006 with the ring never put on.
* Over the moat nothing that walks can reach us: land monsters can't enter water, and mnexto/enexto only put a
  covetous lich on a square it can stand on. The channel along map row 0 (x 1-8) has only undiggable stone
  above it and the NW tower's wall below it -- no land square next to it at all.
* The west maze touches the moat ring's west column (map x 0, rows 0-5 and 11-16) along its whole length, and the
  maze walls outside the castle map are diggable (castle.des NON_DIGGABLE covers the map only). So from any
  landing spot a straight dig reaches (-1,2) [north] or (-1,14) [south], one diagonal float from (0,1) / (0,15):
  9 water squares to the dry strip instead of the 14 from the courtyard corner, and no walk through the maze to
  the courtyard (where 34 of 38 castle minotaur first hits came, castle-breach F055).
* Horizontal digging works while levitating (apply.c use_pick_axe2 checks Levitation only for u.dz != 0).
* An amulet of magical breathing makes us Amphibious (youprop.h): stepping into the moat is 'You fall into the water.
  But you aren't drowning. You touch bottom.' (trap.c drown) and we walk on along the moat bottom -- a lasting way
  round the ring that no castle_logic path used (0 of 6 MB-kit harness games crossed: the amulet was only put on
  against eel wraps). Amulets say nothing when put on, so an unknown one is tested by that very step from the dug
  launch square: without magical breathing drown() lets us crawl back out onto it ('Pheew! That was close.'),
  wet but alive. A breathless or amphibious polymorph form is Amphibious the same way (mondata.h amphibious()).
  Under water each move may be turned aside ('Water turbulence', mkmaze.c water_friction: 1 in 3), and climbing
  out needs at most Burdened (hack.c domove); sea monsters still bite, but an eel's wrap can't drown us.

CFP_RUSH: on the castle's west side, not yet on the moat ring, as a preempt above fight2 / elbereth_rest / the scare
hold / deep_poly_escape (below castle_logic's crossing strategy and the emergency layer):
  1. not floating: put on / wear what may be a LASTING lift right away -- a known ring of levitation or
     levitation/water walking boots, then each unknown ring, then (CFP_MB) unknown amulets that may be magical
     breathing, then unknown boots of a magic appearance (the kit's potions stay for castle_logic: a timed lift is
     quaffed where the crossing starts);
  2. floating (or amphibious, or wearing an amulet still to be water-tested): dig/float straight to the launch
     square and on onto the moat, attacking only what blocks the way (no throwing: a levitating thrower hurtles
     backwards, castle-c4 seed 1 'killed by bumping into a wall').
castle_logic's crossing strategy takes over as soon as we are on the moat ring (castle.committed()).
CFP_ZAP: a sea monster (or anything) blocking the way round is zapped with the best known wand first: teleportation
(u_teleport_mon works on the noteleport castle, F055), then striking (a beam: no bounce), then sleep/fire/cold/
lightning/magic missile rays only along a stretch long enough that the ray can't bounce back at us.
"""

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from .castle_logic import OUTSIDE, WEST_COURTYARD, map_char, to_bot
from .glyph import G
from .strategy import Strategy

# launch squares (maze side, dug if need be) and the first moat square beyond them
NORTH_LAUNCH, NORTH_ENTRY = (-1, 2), (0, 1)
SOUTH_LAUNCH, SOUTH_ENTRY = (-1, 14), (0, 15)
MAGIC_BOOTS = ("combat boots", "jungle boots", "hiking boots", "mud boots", "buckled boots", "riding boots",
               "snow boots")
M1_SWIM, M1_AMPHIBIOUS, M1_BREATHLESS = 0x2, 0x200, 0x400
MB_NAME = 'amulet of magical breathing'   # (objects' full name, power.MB_AMULET)
DIRS8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _log(castle, msg):
    castle.agent.log(f'CFP {msg}')


def _mb(castle):
    """amulet glyph -> 'yes' / 'no': what the water test said about magical breathing."""
    d = getattr(castle, '_cfp_mb', None)
    if d is None:
        d = castle._cfp_mb = {}
    return d


def worn_amulet(castle):
    return next((i for i in castle._items() if i.category == nh.AMULET_CLASS and i.equipped), None)


def amphibious(castle):
    """Magical breathing: a known amulet of magical breathing worn, an amulet whose water test said so, or an
    amphibious / breathless / swimming polymorph form (mondata.h amphibious(), youprop.h Amphibious)."""
    if not jf_config.CFP_MB:
        return False
    a = worn_amulet(castle)
    if a is not None:
        if a.is_unambiguous() and a.object.name == MB_NAME:
            return True
        if _mb(castle).get(a.glyphs[0]) == 'yes':
            return True
    form = form_permonst(castle.agent)
    return form is not None and bool(form.mflags1 & (M1_SWIM | M1_AMPHIBIOUS | M1_BREATHLESS))


def mb_test_pending(castle):
    """An amulet is worn whose magical breathing is still unknown (put on by the lift phase)."""
    if not jf_config.CFP_MB:
        return False
    a = worn_amulet(castle)
    return a is not None and not a.is_unambiguous() and MB_NAME in _names(a) and \
        a.glyphs[0] not in _mb(castle)


def _west_phase(castle):
    """On the castle, west of the moat, not yet on the ring (castle_logic's crossing strategy has it there)."""
    if not castle.active() or castle.committed():
        return False
    mx, my = castle._pos()
    if mx >= 57:
        return False
    return (mx, my) not in OUTSIDE or (mx, my) in WEST_COURTYARD


def _names(item):
    return {getattr(o, 'name', '') for o in item.objs}


def lift_candidates(castle):
    """[(kind, item)] that may be a LASTING lift, best first: known levitation ring / boots, water walking boots,
    a known amulet of magical breathing, unknown rings that may be levitation, unknown amulets that may be magical
    breathing (CFP_MB), unknown boots of a magic appearance. Tested unknown ones are skipped."""
    out = []
    items = castle._items()
    amulet_worn = worn_amulet(castle) is not None
    for it in items:
        if it.category == nh.RING_CLASS and not it.equipped and it.is_unambiguous() and \
                it.object.name == 'levitation':
            out.append((0, 'ring', it))
        elif it.is_armor() and not it.equipped and it.is_unambiguous() and \
                it.object.name in ('levitation boots', 'water walking boots'):
            out.append((1, 'boots', it))
        elif jf_config.CFP_MB and it.category == nh.AMULET_CLASS and not it.equipped and not amulet_worn and \
                it.is_unambiguous() and it.object.name == MB_NAME:
            out.append((1, 'amulet', it))
        elif it.category == nh.RING_CLASS and not it.equipped and not it.is_unambiguous() and \
                'levitation' in _names(it) and it.glyphs[0] not in castle._tested:
            out.append((2, 'ring', it))
        elif jf_config.CFP_MB and it.category == nh.AMULET_CLASS and not it.equipped and not amulet_worn and \
                not it.is_unambiguous() and MB_NAME in _names(it) and \
                it.glyphs[0] not in castle._tested and it.glyphs[0] not in _mb(castle):
            out.append((3, 'amulet', it))
        elif it.is_armor() and not it.equipped and not it.is_unambiguous() and \
                it.glyphs[0] not in castle._tested and \
                ({'levitation boots', 'water walking boots'} & _names(it) or
                 any(b in it.text for b in MAGIC_BOOTS)):
            out.append((4, 'boots', it))
    out.sort(key=lambda t: t[0])
    if any(k == 'amulet' and not i.is_unambiguous() for _, k, i in out):
        # an unknown amulet is tested by a dunk in the moat, which dilutes every potion we carry (trap.c drown ->
        # water_damage_chain; cfp-g1-arm cfpi-s1's diluted potion of levitation became water): the potions go first.
        # And 13.5% of amulets strangle, 90% of those cursed: only with a prayer in hand, or once nothing else is left
        # (harness cfp-cra-on jf14-s6 'killed by strangulation' 16 turns after the lift phase put one on)
        others = [t for t in out if not (t[1] == 'amulet' and not t[2].is_unambiguous())]
        if potion_candidates(castle) or not (_can_pray(castle) or not others and not _poly_hope(castle)):
            out = others
    return [(k, i) for _, k, i in out]


def potion_candidates(castle):
    """Potions castle_logic's plan would quaff for a lift (known levitation first, then unknown ones by P)."""
    try:
        return [item for kind, item in castle._plan() if kind == 'potion']
    except Exception:
        return []


def _poly_hope(castle):
    try:
        from . import castle_power
        return jf_config.CASTLE_POLY and castle_power._poly_wand(castle.agent) is not None
    except Exception:
        return False


def pending(castle):
    """CFP_RUSH still has a lasting lift to try or a water test to make here (castle_power's self-polymorph waits:
    a big form tears the armour off -- cfp-p2-jf25 s3 lost cloak, shield, helm and boots to a leocrotta form,
    then a lynx killed the naked dwarf -- and the kit's own lift or magical breathing is the better bet)."""
    if not jf_config.CFP_RUSH or not _west_phase(castle):
        return False
    if castle._floating():
        return False
    return bool(lift_candidates(castle)) or mb_test_pending(castle)


def _launch(castle):
    """The launch square and moat entry of the nearer half. castle_logic's way round (_half) is set to match: only
    its courtyard (_pick_half) ever set it, so a south launch floated on with _half 'north' -- the south squares are
    off the NORTH bfs, the OUTSIDE bfs ties row 15 (moat along the castle's south wall) with row 16 (the dry strip),
    and _downhill's direction order picks the diagonal first: harness cfp-c1-on seed 3 drifted into row 15 again and
    again and a xorn in the wall (row 14) hit it there each time ('There is a pool of water here.  The xorn hits!'),
    until it died at (34,15). ROUTE_S keeps the strip; nothing that walks next to row 16 but sea monsters."""
    mx, my = castle._pos()
    north = my <= 8
    castle._half = 'north' if north else 'south'
    return (NORTH_LAUNCH, NORTH_ENTRY) if north else (SOUTH_LAUNCH, SOUTH_ENTRY)


def _dig(castle, n):
    """Apply the digging tool at the neighbouring square n (map coords). False: no tool."""
    agent = castle.agent
    tool = castle.dive.digging_tool()
    if tool is None:
        return False
    y, x = to_bot(*n)
    d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
    with agent.atom_operation():
        tool = agent.inventory.move_to_inventory(tool)
        agent.step(A.Command.APPLY)
        agent.type_text(agent.inventory.items.get_letter(tool))
        if 'In what direction do you want to dig?' in agent.single_message:
            agent.direction(d)
        elif agent.single_message.startswith('In what direction'):
            agent.step(A.Command.ESC)
    _log(castle, f'dig {d} at {n}: {agent.message[:90]!r}')
    if 'through thin air' in agent.message:
        # an open square the map memory hadn't marked walkable (castle_logic._dig_toward, pwc-dp3 jf27-s0)
        agent.current_level().walkable[y, x] = True
        agent.last_bfs_step = -1
    if "can't hold it" in agent.message or 'You are unable to' in agent.message or "can't dig" in agent.message:
        return False   # a polymorph form too weak for the pick (cfp-m1-arm jf25-s1: 'You can't hold it strongly enough.')
    castle._tries.pop('wielded', None)   # the pick is in hand now: castle_logic re-wields the weapon once
    return True


def _toward(castle, goal, then):
    """One step or one dig on a straight path to goal (rows first, then columns), then onto `then` (the moat).
    None when stuck (no tool, or the same square resisted 6 digs)."""
    agent = castle.agent
    pos = castle._pos()
    if pos == goal:
        if then is not None and castle._monster_at(*then) and castle._wield_weapon():
            # the dig that opened this square left the pick-axe in hand (_dig clears 'wielded'): castle-c1 seeds 10, 11
            # and 13 fought the shark/eel sitting on the entry square with it -- 'You begin bashing monsters with your
            # pick-axe.  You miss the shark.' five to ten times while it bit them from 90 to 19 HP -- with Excalibur
            # in the pack
            return True
        castle._set_state(f'cfp: launching from {goal} onto {then}')
        castle._step_to(*then)
        return True
    mx, my = int(pos[0]), int(pos[1])   # (numpy ints: numpy bools don't subtract)
    sx = int(goal[0] > mx) - int(goal[0] < mx)
    sy = int(goal[1] > my) - int(goal[1] < my)
    # rows first, then columns -- but never onto the moat or the castle's own (undiggable) structure: from the west
    # courtyard the rows-first step toward the north launch is the moat column x 0 (harness cfp-xk2-arm jf16-s0~5
    # dug 'through thin air' at (0,5) and walked into the water); the maze (x < 0) and the courtyard are fine
    options = [(mx, my + sy)] if sy else []
    if sx:
        options.append((mx + sx, my))
    ok = [o for o in options if map_char(*o) in (' ', '.') and not _wet(castle, *o)]
    if not ok:
        return None
    n = ok[0]
    y, x = to_bot(*n)
    if castle._monster_at(*n):
        if not castle._tries.get('wielded'):
            castle._tries['wielded'] = 1
            if agent.wield_best_melee_weapon():
                return True
        castle._set_state(f'cfp: attacking what blocks {n}')
        with agent.atom_operation():
            agent.step(A.Command.FIGHT)
            agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
        return True
    level = agent.current_level()
    if agent.glyphs[y, x] in G.BOULDER or level.objects[y, x] in G.BOULDER:
        # no leverage to push it while levitating: break it with the pick (dig.c: 'The boulder falls apart')
        key = ('cfp_boulder', n)
        if castle._tries.get(key, 0) >= 4 or not _dig(castle, n):
            return None
        castle._tries[key] = castle._tries.get(key, 0) + 1
        return True
    if level.walkable[y, x]:
        castle._set_state(f'cfp: to the moat via {goal}')
        castle._step_to(*n)
        return True
    key = ('cfp_dig', n)
    if castle._tries.get(key, 0) >= 6:
        return None
    castle._tries[key] = castle._tries.get(key, 0) + 1
    castle._set_state(f'cfp: digging toward {goal}')
    return True if _dig(castle, n) else None


def _water_test_result(castle, before_pos):
    """After the step onto the moat with an amulet on test: under water (on a moat square) = magical breathing;
    crawled out = not. Records the answer; returns it or None if the step didn't happen."""
    a = worn_amulet(castle)
    if a is None:
        return None
    agent = castle.agent
    pos = castle._pos()
    msg = agent.message or ''
    if map_char(*pos) == '}' and not castle._dry(*pos):
        _mb(castle)[a.glyphs[0]] = 'yes'
        _log(castle, f'magical breathing: {a.text!r} took us under water at {pos} ({msg[:80]!r})')
        return 'yes'
    if 'crawl out' in msg or 'Pheew' in msg or 'sink like' in msg:
        _mb(castle)[a.glyphs[0]] = 'no'
        _log(castle, f'no magical breathing: {a.text!r} ({msg[:80]!r}); back at {pos}')
        castle._tested.add(a.glyphs[0])
        return 'no'
    return None


def _step_toward(castle, goal):
    """One step or one dig toward goal (rows first, then columns). 'here' at the goal, None when stuck."""
    if castle._pos() == goal:
        return 'here'
    return _toward(castle, goal, None)


def _on_foot_blocked(castle):
    """On foot (a water test or a potion to quaff): leave it to the usual layers (Elbereth, fight2, the rest) while a
    land hostile is within 2, HP is below 60%, or we are confused/stunned (a confused step next to the moat drowns) --
    a depth-29 lynx took cfp-mb4-jf25 s3 from 72 to 9 HP in 4 turns while the rush fought it off Elbereth."""
    agent = castle.agent
    bl = agent.blstats
    prop = agent.character.prop
    return bool(_land_hostiles_within(castle, 2)) or bl.hitpoints < 0.6 * bl.max_hitpoints or \
        prop.confusion or prop.stun


def rush_strategy(dive):
    """CFP_RUSH (see the module docstring)."""
    def plan():
        castle = dive.castle
        if not jf_config.CFP_RUSH or not _west_phase(castle):
            return None
        pos = castle._pos()
        floating = castle._floating()
        if jf_config.CFP_INVIS and not castle._tries.get('cfp_invis_done') and invis_source(castle) is not None:
            return ('invis',)
        if not floating:
            cands = lift_candidates(castle)
            if cands:
                return ('lift',) + cands[0]
            if _poly_now(castle):
                return ('poly',)
            if jf_config.CFP_PRUSH:
                pots = potion_candidates(castle)
                if pots and castle._tries.get('cfp_potion_stuck', 0) < 3:
                    return None if _on_foot_blocked(castle) else ('potion', pots[0])
            if not mb_test_pending(castle):
                return None
            if _on_foot_blocked(castle):
                return None
        if castle._tries.get('cfp_stuck', 0) >= 3:
            return None
        if pos in WEST_COURTYARD and floating:
            return None   # the courtyard: castle_logic floats us on from here (committed)
        return ('rush',)

    def f():
        castle = dive.castle
        agent = dive.agent
        p = plan()
        if p is None:
            yield False
            return
        yield True
        steps = 0
        last = None
        same = 0
        while p is not None and steps < 300:
            steps += 1
            before = agent.step_count
            if p[0] == 'lift':
                _, kind, item = p
                _log(castle, f'arrival lift: trying {kind} {item.text!r} at {castle._pos()} '
                             f'hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints}')
                if kind == 'ring' and _keep_for_poly(castle, item):
                    # a ring that may be polymorph control stays on for the self-zap to come (CFP_XORN): the lift test
                    # took it off again, and the deep escape zapped blind (harness cfp-xorn1 seed 0: random forms)
                    _put_on_keep(castle, item)
                else:
                    castle._try(kind, item)
                if castle._floating():
                    _log(castle, f'floating ({kind} {item.text!r})')
            elif p[0] == 'invis':
                _invis_step(castle)
            elif p[0] == 'poly':
                _poly_step(castle)
            elif p[0] == 'potion':
                if _potion_step(castle, p[1]) is None:
                    castle._tries['cfp_potion_stuck'] = castle._tries.get('cfp_potion_stuck', 0) + 1
                    _log(castle, f'potion rush stuck at {castle._pos()}')
                    return
            elif _melee_adjacent(castle):
                # something next to us on land: fight it with the real weapon (the dig leaves the pick in hand, and
                # cfp-mb2-jf25 s3 bashed a lynx with it -- 'You begin bashing monsters with your pick-axe' -- missing 8
                # of 11 swings from 77 HP to dead); a faster monster only gets free hits if we walk on
                p = plan()
                last = None
                continue
            else:
                testing = mb_test_pending(castle) and not castle._floating()
                launch, entry = _launch(castle)
                if testing and castle._pos() == launch and \
                        agent.blstats.carrying_capacity > 1 and not castle._tries.get('cfp_mb_unload'):
                    # climbing out of water with magical breathing needs Burdened at most (hack.c domove)
                    castle._tries['cfp_mb_unload'] = 1
                    _log(castle, f'encumbrance {agent.blstats.carrying_capacity} before the water test: unloading')
                    _unload(castle)
                    p = plan()
                    continue
                pos_before = castle._pos()
                r = _toward(castle, launch, entry)
                if r is None:
                    castle._tries['cfp_stuck'] = castle._tries.get('cfp_stuck', 0) + 1
                    _log(castle, f'rush stuck at {castle._pos()} toward {launch}')
                    return
                if testing and pos_before == launch:
                    _water_test_result(castle, pos_before)
            if agent.step_count == before:
                agent.search()
            pos = castle._pos()
            same = same + 1 if pos == last else 0
            last = pos
            if same >= 25:
                castle._tries['cfp_stuck'] = castle._tries.get('cfp_stuck', 0) + 1
                _log(castle, f'rush: no progress at {pos}')
                return
            p = plan()

    return Strategy(f)


def _potion_step(castle, item):
    """CFP_PRUSH: a potion that may lift us is quaffed on the square next to the launch square, not in the courtyard:
    dig the launch square first, stand one square west of it (no water next to it), write Elbereth (a potion of sleeping
    or paralysis leaves us helpless), quaff. A lift then takes two moves to the moat and 9 water squares to the strip
    (from the courtyard's test square: 14), and the walk to the courtyard through the maze is saved. True: acted;
    None: stuck."""
    agent = castle.agent
    launch, entry = _launch(castle)
    test = (launch[0] - 1, launch[1])
    ly, lx = to_bot(*launch)
    level = agent.current_level()
    if not level.walkable[ly, lx]:
        if castle._pos() != test:
            r = _step_toward(castle, test)
            return None if r is None else True
        key = ('cfp_dig', launch)
        if castle._tries.get(key, 0) >= 6:
            return None
        castle._tries[key] = castle._tries.get(key, 0) + 1
        castle._set_state(f'cfp: digging the launch square {launch}')
        return True if _dig(castle, launch) else None
    if castle._pos() != test:
        r = _step_toward(castle, test)
        return None if r is None else True
    engraving = (agent.inventory.engraving_below_me or '').lower()
    if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind and \
            castle._tries.get('cfp_potion_elbereth', 0) < 6:
        castle._tries['cfp_potion_elbereth'] = castle._tries.get('cfp_potion_elbereth', 0) + 1
        castle._set_state(f'cfp: Elbereth before quaffing at {test}')
        agent.engrave('Elbereth')
        return True
    _log(castle, f'potion rush: quaffing {item.text!r} at {test} hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints}')
    castle._try('potion', item)
    if castle._floating():
        _log(castle, f'floating on {item.text!r}: to the moat via {launch}')
    return True


def _land_hostiles_within(castle, r):
    agent = castle.agent
    bl = agent.blstats
    return [m for m in agent.get_visible_monsters()
            if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= r and not _wet(castle, *_to_map(m[1], m[2]))]


def _adjacent_land_hostiles(castle):
    return _land_hostiles_within(castle, 1)


def _melee_adjacent(castle):
    """A hostile next to us on dry land (not the moat's eels and sharks): wield the real weapon (once after each dig)
    and hit it. True: acted. (Only while floating: on foot the usual layers fight.)"""
    agent = castle.agent
    bl = agent.blstats
    if not castle._floating():
        return False
    near = _adjacent_land_hostiles(castle)
    if not near:
        return False
    if not castle._tries.get('wielded'):
        castle._tries['wielded'] = 1
        if agent.wield_best_melee_weapon():
            _log(castle, f'wielded the melee weapon against {getattr(near[0][3], "mname", "?")}')
            return True
    _, y, x, mon, _ = near[0]
    castle._set_state(f'cfp: fighting {getattr(mon, "mname", "?")} next to us')
    with agent.atom_operation():
        agent.step(A.Command.FIGHT)
        agent.direction(agent.calc_direction(bl.y, bl.x, y, x))
    return True


def _to_map(y, x):
    from .castle_logic import to_map
    return to_map(y, x)


def _wet(castle, mx, my):
    """A moat square that is still water (castle_logic._dry says False for every maze square: off the map)."""
    return map_char(mx, my) == '}' and not castle._dry(mx, my)


def _unload(castle):
    """Drop what is heavy and not needed for the crossing (armour worn, weapons wielded, the digging tool, wands,
    rings, amulets, keys and food stay) so that we can climb out of the water again."""
    agent = castle.agent
    keep_names = ('skeleton key', 'lock pick', 'credit card', 'pick-axe', 'dwarvish mattock')
    drop = []
    for it in castle._items():
        if it.equipped or it.category in (nh.COIN_CLASS, nh.WAND_CLASS, nh.RING_CLASS, nh.AMULET_CLASS,
                                          nh.FOOD_CLASS):
            continue
        if any(n in it.text for n in keep_names):
            continue
        if it.category in (nh.WEAPON_CLASS, nh.ARMOR_CLASS, nh.TOOL_CLASS, nh.ROCK_CLASS, nh.GEM_CLASS,
                           nh.BALL_CLASS, nh.CHAIN_CLASS):
            drop.append(it)
    if drop:
        try:
            agent.inventory.drop(drop, smart=False)
        except Exception as e:   # (an inventory quirk must not end the passage)
            _log(castle, f'unload failed: {e!r}')


# ------------------------------------------------------------------------------------------------ CFP_ZAP

BEAMS = ('teleportation', 'striking')
RAYS = ('sleep', 'lightning', 'fire', 'cold', 'magic missile')


def _free_run(castle, d):
    """Squares a ray can travel from us in direction d over the castle map's open squares (moat, floor) before it
    hits a wall or leaves the map (where it would bounce)."""
    dx = (1 if 'e' in d else -1 if 'w' in d else 0)
    dy = (1 if 's' in d else -1 if 'n' in d else 0)
    mx, my = castle._pos()
    n = 0
    while n < 20:
        mx, my = mx + dx, my + dy
        if map_char(mx, my) not in '.}':
            break
        n += 1
    return n


def blocker_zap(castle, mx, my, d):
    """CFP_ZAP: zap the best known wand at what blocks (mx, my) in direction d. True: zapped."""
    if not jf_config.CFP_ZAP:
        return False
    key = ('cfp_zap', (mx, my))
    t = castle._tries
    if t.get(key, 0) >= 5:
        return False
    items = castle._items()
    choice = None
    for name in BEAMS:
        choice = next((i for i in items if castle._usable_wand(i, name)), None)
        if choice is not None:
            break
    if choice is None and _free_run(castle, d) >= 14:
        # rays travel rn1(7,7) squares and bounce off walls: only along a stretch that long (zap.c dobuzz)
        for name in RAYS:
            choice = next((i for i in items if castle._usable_wand(i, name)), None)
            if choice is not None:
                break
    if choice is None and not t.get(('cfp_vanish', (mx, my))):
        # last resort: an unknown wand that the engrave test left as cancellation / teleportation / make invisible
        # (power_route.vanish_wand): 1 in 3 it sends the blocker elsewhere on the level (u_teleport_mon works on the
        # noteleport castle); a demilich form spent 7 moat turns on a shark at (0,3) with such a wand unused
        # (power-route pr-a25s1-m1). Once per blocker square.
        try:
            from . import power_route
            choice = power_route.vanish_wand(castle.agent)
        except Exception:
            choice = None
        if choice is not None:
            t[('cfp_vanish', (mx, my))] = 1
    if choice is None:
        return False
    t[key] = t.get(key, 0) + 1
    castle._set_state(f'cfp: zapping {choice.text!r} at what blocks {(mx, my)}')
    castle.agent.zap(choice, d)
    _log(castle, f'zap {choice.text!r} {d} at {(mx, my)}: {castle.agent.message[:100]!r}')
    if 'Nothing happens' in castle.agent.message or 'You wrest' in castle.agent.message:
        castle.agent.inventory.empty_wands.add(choice.text)
    return True


# ------------------------------------------------------------------------------------------------ CFP_XORN

M1_WALLWALK = 0x8
# rooms with monsters in castle.des (map coords, inclusive): antechamber (8 soldiers + lieutenant), the barracks
# (fill_zoo soldiers), the throne room (27 court monsters), the four towers (2 soldiers each), the dragons' alcoves
DANGER_ROOMS = ((7, 5, 14, 11), (16, 5, 25, 6), (16, 10, 25, 11), (27, 5, 37, 11), (2, 2, 6, 3), (56, 2, 60, 3),
                (2, 13, 6, 14), (56, 13, 60, 14), (47, 5, 47, 6), (47, 10, 47, 11))
TRAPDOORS = ((40, 8), (44, 8), (48, 8), (52, 8), (55, 8))


def wallwalker(agent):
    """Our polymorph form walks through walls (M1_WALLWALK: a xorn). The form comes from the messages first
    (note_message): an invisible hero's square shows no monster glyph."""
    form = form_permonst(agent)
    return form is not None and bool(form.mflags1 & M1_WALLWALK)


def _in_room(mx, my, pad=0):
    return any(x0 - pad <= mx <= x1 + pad and y0 - pad <= my <= y1 + pad for x0, y0, x1, y1 in DANGER_ROOMS)


def _xorn_path(castle, start):
    """Cheapest way for a wall-walker from start to a trap door (40..55,08): through walls, doors and rock, around the
    rooms with monsters (a xorn in the wall beside a room is in reach of it) and well away from the moat. The solid
    stone above and below the castle map (level rows 1-2 and 20, map y -2..-1 and 17; bound_digging makes it
    undiggable, not unpassable: hack.c may_passwall only refuses W_NONPASSWALL, which castle.des never sets) runs the
    whole width with nothing that walks beside it, so the way east is: up into that stone, along it, and down at
    x 40 across one moat square (row 1), the castle wall, the north corridor, a storeroom and its wall onto the trap
    door. (Harness cfp-xk1-arm: xorn forms that took the moat's row 16 lost 28 of 30 HP to sharks in 6 turns.)"""
    import heapq
    agent = castle.agent
    mons = set()
    for m in agent.get_visible_monsters():
        mons.add(tuple(castle_to_map(m[1], m[2])))

    def cost(p):
        mx, my = p
        c = 1
        if _in_room(mx, my):
            c += 40
        elif _in_room(mx, my, pad=1):
            c += 12
        if map_char(mx, my) == '}':
            c += 20
        elif any(map_char(mx + dx, my + dy) == '}' for dx, dy in DIRS8):
            # beside the moat: its sharks and eels bite into the wall (harness cfp-xorn1 lost 36 of 44 form HP along
            # row 2, next to the moat's row 1; the corridor on row 3 has walls on both sides)
            c += 4
        if p in mons:
            c += 60
        elif any((mx + dx, my + dy) in mons for dx, dy in DIRS8):
            c += 8
        return c

    goals = set(TRAPDOORS)
    dist = {start: 0}
    prev = {}
    heap = [(0, start)]
    while heap:
        d, p = heapq.heappop(heap)
        if d > dist.get(p, 1 << 30):
            continue
        if p in goals:
            path = [p]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return path[::-1]
        for dx, dy in DIRS8:
            n = (p[0] + dx, p[1] + dy)
            if not (-8 <= n[0] <= 70 and -2 <= n[1] <= 17):
                continue
            nd = d + cost(n)
            if nd < dist.get(n, 1 << 30):
                dist[n] = nd
                prev[n] = p
                heapq.heappush(heap, (nd, n))
    return None


def castle_to_map(y, x):
    from .castle_logic import to_map
    return to_map(y, x)


def xorn_strategy(dive):
    """CFP_XORN: as a wall-walking form on the castle, walk straight through the walls to a trap door and drop into the
    Valley (trap.c fall_through: Is_stronghold -> find_hell; a xorn neither flies nor floats). A controlled polymorph
    (POLY_XORN answers 'Become what kind of monster?' with 'xorn') gets there; castle.des has no NON_PASSWALL."""
    def f():
        castle = dive.castle
        agent = dive.agent
        if not jf_config.CFP_XORN or castle.castle_key is None or \
                agent.current_level().key() != castle.castle_key or not wallwalker(agent):
            yield False
            return
        yield True
        steps = 0
        last = None
        same = 0
        while steps < 600 and wallwalker(agent) and agent.current_level().key() == castle.castle_key:
            steps += 1
            pos = castle._pos()
            path = _xorn_path(castle, pos)
            if not path or len(path) < 2:
                _log(castle, f'xorn: no way to a trap door from {pos}')
                agent.search()
                continue
            nxt = path[1]
            y, x = to_bot(*nxt)
            d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
            if castle._monster_at(*nxt):
                castle._set_state(f'cfp xorn: attacking what blocks {nxt}')
                with agent.atom_operation():
                    agent.step(A.Command.FIGHT)
                    agent.direction(d)
            else:
                if steps == 1 or steps % 10 == 0:
                    _log(castle, f'xorn: at {pos}, {len(path) - 1} steps to {path[-1]} via {path[1:4]}')
                castle._set_state('cfp xorn: walking through the walls to a trap door')
                agent.direction(d)
            pos2 = castle._pos()
            same = same + 1 if pos2 == last else 0
            last = pos2
            if same >= 30:
                _log(castle, f'xorn: stuck at {pos2}')
                return

    return Strategy(f)


def poly_prep(passage, known_only=False):
    """CFP_XORN, before a self-zap with a wand of polymorph: put on the rings that may be polymorph control (unknown
    rings, or a known one) -- with control the game asks 'Become what kind of monster?' and POLY_XORN answers xorn.
    castle_logic's lift test took each unknown ring off again. True: acted this step."""
    if not jf_config.CFP_XORN:
        return False
    agent = passage.agent
    worn = passage._worn_rings()
    if len(worn) >= 2:
        return False
    t = passage._tries
    for it in passage._items():
        if it.category != nh.RING_CLASS or it.equipped or 'polymorph control' not in _names(it):
            continue
        if it.is_unambiguous() and it.object.name != 'polymorph control':
            continue
        if known_only and not it.is_unambiguous():
            continue
        key = ('cfp_polyring', it.glyphs[0])
        if t.get(key):
            continue
        t[key] = 1
        letter = agent.inventory.items.get_letter(it)

        def gen():
            if 'What do you want to put on?' not in agent.single_message:
                return
            yield letter
            if 'Which ring-finger' in agent.single_message:
                yield 'l' if any('left' in r.text for r in worn) is False else 'r'

        with agent.atom_operation():
            agent.step(A.Command.PUTON, gen())
        agent.inventory.items.update(force=True)
        _log(passage, f'xorn prep: put on {it.text!r} (may be polymorph control): {agent.message[:80]!r}')
        return True
    return False


def _keep_for_poly(castle, item):
    """CFP_XORN: a known wand of polymorph is carried and this unknown ring may be polymorph control, with a finger
    free: put it on and leave it on."""
    if not jf_config.CFP_XORN or item.is_unambiguous() or 'polymorph control' not in _names(item):
        return False
    if len(castle._worn_rings()) >= 2:
        return False
    try:
        from . import castle_power
        return castle_power._poly_wand(castle.agent) is not None
    except Exception:
        return False


def _put_on_keep(castle, item):
    """castle_logic._put_on_ring without taking a non-levitation ring off again."""
    agent = castle.agent
    letter = agent.inventory.items.get_letter(item)
    left_busy = any('left' in r.text for r in castle._worn_rings())

    def gen():
        if 'What do you want to put on?' not in agent.single_message:
            return
        yield letter
        if 'Which ring-finger' in agent.single_message:
            yield 'r' if left_busy else 'l'

    with agent.atom_operation():
        agent.step(A.Command.PUTON, gen())
    castle._tested.add(item.glyphs[0])
    castle._tries[('cfp_polyring', item.glyphs[0])] = 1
    agent.inventory.items.update(force=True)
    _log(castle, f'put on {item.text!r} and keep it on (may be polymorph control): {agent.message[:80]!r}')
    if castle.levitating():
        castle._lev_source = ('ring', item.glyphs[0])


# ------------------------------------------------------------------------------------------------ form tracking

import re as _re
_FORM_RX = _re.compile(r"You turn into an? (?:male |female )?([a-z' -]+?)!")


def note_message(agent):
    """agent.update, every observation (CFP_XORN): remember our polymorph form from the messages. castle_power's
    current_form() reads the glyph on our square, which an invisible hero doesn't show (cfp-w1-arm jf16 s0: the kit's
    ring of invisibility went on with the ring of polymorph control; 'You turn into a male xorn!' and the bot, seeing
    no form, zapped the wand again). No steps."""
    msg = agent.single_message or ''
    if not msg:
        return
    try:
        m = _FORM_RX.search(msg)
        if m:
            name = m.group(1).strip()
            if getattr(agent, '_cfp_form', None) != name:
                agent._cfp_form = name
                agent.log(f'CFP form: {name}')
        if "can't see yourself" in msg or 'can see right through yourself' in msg:
            agent._cfp_invis = True
        elif 'body seems to unfade' in msg or 'no longer see through yourself' in msg:
            agent._cfp_invis = False
        if m:
            pass
        elif 'You return to' in msg and 'form' in msg or 'You feel like a new' in msg:
            if getattr(agent, '_cfp_form', None) is not None:
                agent._cfp_form = None
                agent.log('CFP form: back in our own form')
    except Exception as e:   # must never break the step loop
        agent.log(f'CFP note_message failed: {e!r}')


def form_permonst(agent):
    """The form from the messages (CFP_XORN), else castle_power's glyph test."""
    name = getattr(agent, '_cfp_form', None)
    if name:
        try:
            for i in range(nh.NUMMONS):
                p = nh.permonst(i)
                if p.mname == name:
                    return p
        except Exception:
            pass
    try:
        from . import castle_power
        return castle_power.current_form(agent)
    except Exception:
        return None


def _can_pray(castle):
    """A prayer would likely work now (the emergency layer prays away a cursed amulet of strangulation)."""
    agent = castle.agent
    try:
        return not agent.prayer_failed and agent.is_safe_to_pray(500)
    except Exception:
        return False


def door_zap_from_afar(castle, pos):
    """CFP_ZAP: on the east courtyard's row 8, two or more squares east of the locked back door (56,08), zap a known
    wand of striking / digging / opening west at it: zap.c bhit reaches it (range 6-13) and doorlock()/zap_dig break,
    raze or unlock it -- the giant eels start at (57,07)/(57,09), next to the square in front of the door where
    castle_logic zaps, kicks and unlocks (3 of ~40 harness crossings drowned there). True: zapped."""
    if not jf_config.CFP_ZAP:
        return False
    from .castle_logic import DOOR
    mx, my = pos
    if my != 8 or not 58 <= mx <= 62 or castle._door_open():
        return False
    if any(castle._monster_at(x, 8) for x in range(57, mx)):
        return False   # (the beam would stop at it)
    t = castle._tries
    for name in ('striking', 'digging', 'opening'):
        wand = next((i for i in castle._items() if castle._usable_wand(i, name)), None)
        key = ('cfp_doorzap', name)
        if wand is None or t.get(key, 0) >= 2:
            continue
        t[key] = t.get(key, 0) + 1
        castle._set_state(f'cfp: zapping {name} at the back door from {pos}')
        castle.agent.zap(wand, 'w')
        _log(castle, f'door zap {name} from {pos}: {castle.agent.message[:100]!r}')
        return True
    return False


def _poly_now(castle):
    """CFP_XORN: a known wand of polymorph with charges, a worn ring that may be polymorph control, and no wall-walking
    form yet: zap ourselves now (POLY_XORN answers 'xorn'), before the potions -- castle_power's own self-zap waits at
    the bottom of the preempt chain (harness cfp-xk2-arm jf16-s0: the potion rush ran first)."""
    if not jf_config.CFP_XORN or castle._tries.get('cfp_nocontrol') or wallwalker(castle.agent):
        return False
    if castle._tries.get('pw_polyzaps', 0) >= 12:
        return False
    try:
        from . import castle_power
        if castle_power._poly_wand(castle.agent) is None:
            return False
    except Exception:
        return False
    return any('polymorph control' in _names(r) and
               (not r.is_unambiguous() or r.object.name == 'polymorph control') for r in castle._worn_rings())


def _poly_step(castle):
    from . import castle_power
    agent = castle.agent
    wand = castle_power._poly_wand(agent)
    castle._tries['pw_polyzaps'] = castle._tries.get('pw_polyzaps', 0) + 1
    _log(castle, f'xorn: zapping {wand.text!r} at ourselves (zap {castle._tries["pw_polyzaps"]})')
    castle_power._zap_self(castle, wand)
    if 'Become what kind of monster' not in (agent.message or '') and \
            'You feel like a new' not in (agent.message or '') and not wallwalker(agent):
        # no control prompt: none of the worn rings is polymorph control -- leave the wand to CASTLE_POLY's random forms
        castle._tries['cfp_nocontrol'] = 1
        _log(castle, f'xorn: no polymorph control ({(agent.message or "")[:80]!r})')


# ------------------------------------------------------------------------------------------------ CFP_INVIS

def invis_source(castle):
    """A known way to permanent invisibility: a wand of make invisible zapped at ourselves (zap.c zapyourself: 'ordinary'
    -> HInvis |= FROMOUTSIDE) or a ring of invisibility; None when already invisible (tracked from the messages)."""
    agent = castle.agent
    if getattr(agent, '_cfp_invis', False):
        return None
    for it in castle._items():
        if castle._usable_wand(it, 'make invisible'):
            return ('zap', it)
        if it.category == nh.RING_CLASS and not it.equipped and it.is_unambiguous() and \
                it.object.name == 'invisibility' and len(castle._worn_rings()) < 2:
            return ('ring', it)
    return None


def _invis_step(castle):
    """CFP_INVIS: become invisible on the castle before the crossing. A monster that can't see us re-guesses our square
    each time we move (monmove.c set_apparxy: our real square 1 time in 3, else a random accessible square next to us),
    so melee from sharks, eels, minotaurs and soldiers lands ~40-55% as often. A worn mummy wrapping blocks it (BInvis:
    'You feel rather itchy under ...') and comes off first."""
    agent = castle.agent
    t = castle._tries
    src = invis_source(castle)
    if src is None:
        t['cfp_invis_done'] = 1
        return
    wrap = next((i for i in castle._items() if i.is_armor() and i.equipped and 'mummy wrapping' in i.text), None)
    if wrap is not None and not t.get('cfp_unwrap'):
        t['cfp_unwrap'] = 1
        _log(castle, f'invisibility: taking off {wrap.text!r} first')
        if wrap.status != wrap.CURSED:
            agent.inventory.takeoff(wrap)
            return
    t['cfp_invis_done'] = 1
    kind, item = src
    if kind == 'zap':
        _log(castle, f'invisibility: zapping {item.text!r} at ourselves')
        agent.zap(item, '.')
        if 'Nothing happens' in agent.message or 'You wrest' in agent.message:
            agent.inventory.empty_wands.add(item.text)
            t['cfp_invis_done'] = 0   # (another source may be left)
    else:
        letter = agent.inventory.items.get_letter(item)
        left_busy = any('left' in r.text for r in castle._worn_rings())

        def gen():
            if 'What do you want to put on?' not in agent.single_message:
                return
            yield letter
            if 'Which ring-finger' in agent.single_message:
                yield 'r' if left_busy else 'l'

        with agent.atom_operation():
            agent.step(A.Command.PUTON, gen())
        agent.inventory.items.update(force=True)
    _log(castle, f'invisibility: {agent.message[:90]!r}')
