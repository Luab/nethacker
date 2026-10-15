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

import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from .castle_logic import OUTSIDE, WEST_COURTYARD, map_char, to_bot
from .exceptions import AgentChangeStrategy, AgentFinished, AgentPanic
from .glyph import G, SS
from .strategy import Strategy

# launch squares (maze side, dug if need be) and the first moat square beyond them
NORTH_LAUNCH, NORTH_ENTRY = (-1, 2), (0, 1)
SOUTH_LAUNCH, SOUTH_ENTRY = (-1, 14), (0, 15)
MAGIC_BOOTS = ("combat boots", "jungle boots", "hiking boots", "mud boots", "buckled boots", "riding boots",
               "snow boots")
M1_SWIM, M1_AMPHIBIOUS, M1_BREATHLESS = 0x2, 0x200, 0x400
M1_NOHANDS, M1_NOLIMBS = 0x2000, 0x6000   # (monflag.h: no hands to apply a digging tool)
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
    if jf_config.LIFT_NEAR_WATER and castle._floating():
        return _near_launch(castle)
    castle._half = 'north' if north else 'south'
    return (NORTH_LAUNCH, NORTH_ENTRY) if north else (SOUTH_LAUNCH, SOUTH_ENTRY)


def _path_cost(castle, frm, to):
    """Turns for the rush's straight way (rows first, then columns) from frm to to: a known walkable square 1, anything
    else a dig (2.5: a maze wall takes a dwarf 2-3 turns)."""
    agent = castle.agent
    level = agent.current_level()
    x, y = int(frm[0]), int(frm[1])
    cost = 0.0
    while (x, y) != tuple(to):
        if y != to[1]:
            y += 1 if to[1] > y else -1
        else:
            x += 1 if to[0] > x else -1
        by, bx = to_bot(x, y)
        ok = 0 <= by < level.walkable.shape[0] and 0 <= bx < level.walkable.shape[1] and level.walkable[by, bx]
        cost += 1.0 if ok else 2.5
    return cost


def _near_launch(castle):
    """LIFT_NEAR_WATER: the quickest way onto the water for a floating hero, by the estimated turns (digging through the
    maze vs floating over moat squares at ~0.75 turns each with intrinsic speed): the fixed launch squares (-1,2)/(-1,14)
    (9 moat squares to the dry strip), the moat column straight east of us when we are level with it (rows 0-5 / 11-16:
    no vertical dig), or -- level with the courtyard (rows 6-10) -- straight east into the courtyard, whose corner opens
    onto the column (13 moat squares). cmp-main jf42-s2 floated at (-2,10) and dug 4 squares south to (-1,14) through a
    boulder and five fights (43 turns) while the courtyard's join (-1,10) was one step east."""
    mx, my = castle._pos()
    options = []
    for launch, entry, water, half in ((NORTH_LAUNCH, NORTH_ENTRY, 9, 'north'), (SOUTH_LAUNCH, SOUTH_ENTRY, 9, 'south')):
        options.append((_path_cost(castle, (mx, my), launch) + 0.75 * water, launch, entry, half))
    if 0 <= my <= 5 or 11 <= my <= 16:
        half = 'north' if my <= 5 else 'south'
        water = (max(my, 1) + 8) if half == 'north' else (max(16 - my, 1) + 8)
        launch, entry = (-1, my), (0, my)
        options.append((_path_cost(castle, (mx, my), launch) + 0.75 * water, launch, entry, half))
    elif 6 <= my <= 10:
        half = 'north' if my <= 8 else 'south'
        corner = 6 if half == 'north' else 10
        launch, entry = (-1, my), (0, my)
        options.append((_path_cost(castle, (mx, my), launch) + 1 + 0.75 * (abs(my - corner) + 13), launch, entry, half))
    options.sort(key=lambda o: o[0])
    cost, launch, entry, half = options[0]
    castle._half = half
    t = castle._tries
    if t.get('lift_near_logged') != (launch, entry):
        t['lift_near_logged'] = (launch, entry)
        _log(castle, f'near water: from {(int(mx), int(my))} via {launch} onto {entry} ({half}, ~{cost:.0f} turns; '
                     f'options {[(round(o[0]), o[1]) for o in options]})')
    return launch, entry


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
        form = form_permonst(agent)
        if form is not None:
            castle._tries['cfp_nodig'] = form.mname   # LIFT_POLY_PICKY: this form can't dig through the maze
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
            if _swap_for_fight(castle):
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
        if jf_config.CL_ROUTE and not floating and _cl_webbed(castle):
            return ('cl_web',)
        if jf_config.CFP_INVIS and not castle._tries.get('cfp_invis_done') and invis_source(castle) is not None:
            return ('invis',)
        if not floating:
            if jf_config.CL_LAUNCH and jf_config.CL_ROUTE:
                # castle-lift: walk to the far launch square first and try the lifts there on Elbereth (see cl_launch_plan)
                lp = cl_launch_plan(castle)
                if lp is not None:
                    return lp
            cands = lift_candidates(castle)
            if cands:
                return ('lift',) + cands[0]
            if _poly_now(castle):
                return ('poly',)
            if jf_config.CL_POTION_EARLY:
                # castle-lift: a potion that may be levitation, right here at the first quiet moment (see early_potion)
                pot = early_potion(castle)
                if pot is not None:
                    return ('cl_potion', pot)
            if jf_config.CFP_PRUSH and (pos[1] <= 5 or pos[1] >= 11) and pos not in WEST_COURTYARD:
                # (only off the courtyard's rows 6-10: from there castle_logic's walk east to TEST_SPOT is the short way
                # -- harness cfp-real jf25-s0 met a troll digging from the courtyard rows up to (-2,2))
                pots = potion_candidates(castle)
                if pots and castle._tries.get('cfp_potion_stuck', 0) < 3:
                    return None if _on_foot_blocked(castle) else ('potion', pots[0])
            if not mb_test_pending(castle):
                return None
            if _on_foot_blocked(castle):
                return None
        if castle._tries.get('cfp_stuck', 0) >= 3:
            return None
        if pos in WEST_COURTYARD and floating and \
                not (jf_config.CL_ROUTE and not castle.committed() and not castle._tries.get('cl_court_off')):
            return None   # the courtyard: castle_logic floats us on from here (committed)
        if jf_config.LIFT_POLY_PICKY and floating:
            # an eyeless / sessile / non-digging form: castle_power zaps the wand of polymorph again instead
            from . import castle_power
            if castle_power.unusable_form(castle) is not None and castle_power._poly_wand(castle.agent) is not None:
                return None
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
            elif p[0] == 'cl_potion':
                _early_potion_step(castle, p[1])
                if agent.step_count == before:
                    agent.search()
                last = None   # (quaffing in place is not 'no progress': the rush's stuck count stays for the float)
                p = plan()
                continue
            elif p[0] == 'cl_web':
                # CL_ROUTE: a web holds us and hides the lift (see _cl_webbed); castle_logic would give the passage up
                t = castle._tries
                t['cl_web_tries'] = t.get('cl_web_tries', 0) + 1
                _log(castle, f'cl route: held in a web at {castle._pos()}: pulling free ({t["cl_web_tries"]})')
                _cl_web_pull(castle)
                if agent.step_count == before:
                    agent.search()
                last = None
                p = plan()
                continue
            elif p[0] in ('cl_walk', 'cl_elbereth', 'cl_wait'):
                t = castle._tries
                if p[0] == 'cl_walk':
                    t['cl_launch_steps'] = t.get('cl_launch_steps', 0) + 1
                    r = cl_route_step(castle, on_foot=True, goals=p[1])
                    if r is None:
                        t['cl_launch_off'] = 1
                        _log(castle, f'cl launch: no way to the water from {castle._pos()}: lifts tried here')
                    elif r == 'launch':
                        last = None
                elif p[0] == 'cl_elbereth':
                    t['cl_launch_elb'] = t.get('cl_launch_elb', 0) + 1
                    if t['cl_launch_elb'] == 1:
                        _log(castle, f'cl launch: on the launch square {castle._pos()} for the moat at '
                                     f'{t.get("cl_launch")}: Elbereth, then the lifts')
                    castle._set_state('cl launch: Elbereth on the launch square')
                    agent.engrave('Elbereth')
                    last = None
                else:
                    t['cl_launch_rest'] = t.get('cl_launch_rest', 0) + 1
                    castle._set_state('cl launch: resting before the lift tests')
                    agent.search()
                    last = None
                if agent.step_count == before:
                    agent.search()
                if p[0] != 'cl_walk':
                    p = plan()
                    continue
            elif jf_config.CFP_DUEL and _duel_flee(castle):
                p = plan()
                last = None
                continue
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
                if jf_config.CFP_DUEL and not testing and _duel_step(castle, launch, entry):
                    if agent.step_count == before:
                        agent.search()
                    last = None   # (the duel stands on the launch square on purpose: not 'no progress')
                    p = plan()
                    continue
                r = None
                if jf_config.CL_ROUTE and not testing and castle._floating():
                    # castle-lift: the maze-aware way to the cheapest moat entry (see cl_route)
                    r = cl_route_step(castle)
                    if r is None and castle._pos() in WEST_COURTYARD:
                        # no way back out from the courtyard: castle_logic's crossing goes round the corner as before
                        castle._tries['cl_court_off'] = 1
                        _log(castle, f'cl route: none from the courtyard at {castle._pos()}: crossing from here')
                        return
                elif jf_config.CL_ROUTE and jf_config.CL_MB_ROUTE and testing:
                    # castle-lift CL_MB_ROUTE: the magical-breathing water test walks the same maze-aware way on foot and
                    # dunks at the chosen entry (if the amulet works, that entry is where the moat-bottom walk starts)
                    cplan = cl_route(castle)
                    into_water = cplan is not None and map_char(*cplan[0][0]) == '}'
                    if into_water and agent.blstats.carrying_capacity > 1 and not castle._tries.get('cfp_mb_unload'):
                        castle._tries['cfp_mb_unload'] = 1
                        _log(castle, f'encumbrance {agent.blstats.carrying_capacity} before the water test: unloading')
                        _unload(castle)
                        p = plan()
                        continue
                    r = cl_route_step(castle)
                    if r is not None and into_water:
                        _water_test_result(castle, pos_before)
                        r = True
                        pos_before = None
                if r is None:
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


# ------------------------------------------------------------------------------------------------ CFP_DUEL

DUEL_QUIET = 5          # quiet turns on the launch square (no sea monster in reach) before we float out
DUEL_TURNS = 400        # duel + rest turns at most, then we go whatever the state
DUEL_REST_HP = 0.5      # below this (after a fight): step off the water's reach and rest...
DUEL_GO_HP = 0.9        # ...back up to this; the first launch waits for it too
HELD_WORDS = ('swings itself around you', 'cannot escape from')


def _lasting_lift(castle):
    """Floating on what doesn't run out: water walking boots, or a KNOWN levitation ring/boots worn (castle_logic's
    _timed_levitation is False only then). A potion's turns can't be spent on a duel."""
    return castle.water_walking() or (castle.levitating() and not castle._timed_levitation())


def _dry_neighbour(castle, launch):
    """A walkable maze square next to the launch square with no moat next to it (the rush digs in along the launch
    row, so the square west of it is usually open): out of every sea monster's reach."""
    level = castle.agent.current_level()
    for dy in (0, -1, 1):
        sq = (launch[0] - 1, launch[1] + dy)
        if any(map_char(sq[0] + ex, sq[1] + ey) == '}' for ex, ey in DIRS8):
            continue
        y, x = to_bot(*sq)
        if 0 <= y < level.walkable.shape[0] and 0 <= x < level.walkable.shape[1] and level.walkable[y, x]:
            return sq
    return None


def _sea_targets(castle):
    """Hostile monsters (or a remembered unseen one: it just bit us) on the moat squares next to us."""
    agent = castle.agent
    mx, my = castle._pos()
    out = []
    for dx, dy in DIRS8:
        n = (mx + dx, my + dy)
        if not _wet(castle, *n) or not castle._monster_at(*n):
            continue
        y, x = to_bot(*n)
        if agent.monster_tracker.peaceful_monster_mask[y, x]:
            continue
        out.append(n)
    return out


def _duel_hit(castle, n, what):
    """A known beam wand first (CFP_ZAP: teleportation sends it elsewhere on the moat, striking never bounces), else
    the real weapon (the dig leaves the pick in hand)."""
    agent = castle.agent
    y, x = to_bot(*n)
    d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
    if blocker_zap(castle, n[0], n[1], d):
        return
    if not castle._tries.get('wielded'):
        castle._tries['wielded'] = 1
        if agent.wield_best_melee_weapon():
            return
    castle._set_state(f'cfp duel: {what} {n}')
    with agent.atom_operation():
        agent.step(A.Command.FIGHT)
        agent.direction(d)


def _duel_flee(castle):
    """CFP_DUEL: a minotaur in sight while we wait on the launch square (or rest behind it) ends the duel at once --
    the water is the one place it can't follow (46% of castle games meet one, 34 of 38 first hits came in the west
    maze, F055); _melee_adjacent would fight it (3d10/3d10/2d8 a turn). True: acted."""
    d = castle._tries.get('cfp_duel')
    if not d or d['turns'] >= DUEL_TURNS or not _lasting_lift(castle):
        return False
    launch, entry = _launch(castle)
    spot = _dry_neighbour(castle, launch)
    pos = castle._pos()
    if pos not in (launch, spot):
        return False
    if not any(getattr(m[3], 'mname', '') == 'minotaur' for m in _land_hostiles_within(castle, 6)):
        return False
    _log(castle, f'duel: a minotaur in sight at {pos}: out onto the water')
    d['turns'] = DUEL_TURNS
    d['rest'] = False
    castle._step_to(*(launch if pos == spot else entry))
    return True


def _duel_step(castle, launch, entry):
    """CFP_DUEL (see jf_config): on the launch square or the dry square behind it, floating on a lasting lift. The
    sea monsters come to us (they track us and wait at the water square nearest us; a hidden one shows itself with
    its first bite, mhitu.c: 'Wait, <you>! There's a shark hiding under the water!', and stays in view while it
    fights), so: hit whatever is in the water next to us, step back out of reach below half HP and rest (nothing
    in the water reaches that square, no xorn comes out of the castle), and float out only at 90% HP once the water
    next to us has been quiet for DUEL_QUIET turns. True: acted this step (not launching yet)."""
    agent = castle.agent
    bl = agent.blstats
    if not _lasting_lift(castle):
        return False
    d = castle._tries.setdefault('cfp_duel', {'turns': 0, 'quiet': 0, 'rest': False, 'rests': 0, 'fights': 0,
                                              'held': None, 'digs': 0})
    if d['turns'] >= DUEL_TURNS:
        return False
    pos = castle._pos()
    spot = _dry_neighbour(castle, launch)
    if pos != launch and not (d['rest'] and pos == spot):
        return False   # (on the way in _toward walks/digs to the launch square: the square west of it comes first)
    d['turns'] += 1
    msg = agent.message or ''
    if any(w in msg for w in HELD_WORDS):
        d['held'] = bl.time
    targets = _sea_targets(castle) if pos == launch else []
    if d['held'] is not None and bl.time - d['held'] <= 3 and targets:
        # a giant eel's wrap drowns with its next touch, on land or floating (mhitu.c AD_WRAP): hit it, don't walk
        _duel_hit(castle, targets[0], 'held: hitting')
        return True
    if d['rest']:
        if bl.hitpoints >= DUEL_GO_HP * bl.max_hitpoints or d['turns'] >= DUEL_TURNS - 1:
            d['rest'] = False
            d['quiet'] = 0
            _log(castle, f'duel: rested to {bl.hitpoints}/{bl.max_hitpoints}, back to {launch}')
            castle._step_to(*launch)
            return True
        if pos != spot:
            castle._set_state(f'cfp duel: off the water to rest at {spot}')
            castle._step_to(*spot)
            return True
        castle._set_state(f'cfp duel: resting at {spot} ({bl.hitpoints}/{bl.max_hitpoints})')
        agent.search(3)
        return True
    # on the launch square
    low = DUEL_REST_HP if d['fights'] else DUEL_GO_HP
    if bl.hitpoints < low * bl.max_hitpoints and d['rests'] < 4 and not targets:
        if spot is None:
            # no dry square behind us yet: dig the one west of the launch square (the maze outside the castle map is
            # diggable), a few tries; failing that rest right here, the water next to us watched
            if d['digs'] < 6 and _dig(castle, (launch[0] - 1, launch[1])):
                d['digs'] += 1
                return True
            castle._set_state(f'cfp duel: resting on the launch square ({bl.hitpoints}/{bl.max_hitpoints})')
            agent.search()
            return True
        d['rest'] = True
        d['rests'] += 1
        _log(castle, f'duel: {bl.hitpoints}/{bl.max_hitpoints} at {pos}, resting at {spot}')
        castle._step_to(*spot)
        return True
    if targets:
        if bl.hitpoints < DUEL_REST_HP * bl.max_hitpoints and spot is not None and d['rests'] < 4:
            # (a sea monster in reach and we are low: out of its reach first, it can't follow)
            d['rest'] = True
            d['rests'] += 1
            _log(castle, f'duel: {bl.hitpoints}/{bl.max_hitpoints} with {targets} in reach, resting at {spot}')
            castle._step_to(*spot)
            return True
        d['fights'] += 1
        d['quiet'] = 0
        entry_first = sorted(targets, key=lambda n: n != entry)
        _duel_hit(castle, entry_first[0], 'hitting the sea monster at')
        return True
    if any(w in msg for w in castle.SEA_WORDS) or 'hiding under' in msg or 'It bites' in msg:
        d['quiet'] = 0
    else:
        d['quiet'] += 1
    if d['quiet'] > DUEL_QUIET:
        _log(castle, f'duel: quiet at {pos} with {bl.hitpoints}/{bl.max_hitpoints} after {d["fights"]} blows, '
                     f'{d["rests"]} rests, {d["turns"]} turns: launching')
        d['turns'] = DUEL_TURNS   # (once out, no second duel if a step back brings us here again)
        return False
    castle._set_state('cfp duel: waiting on the launch square for what hides in the water')
    agent.search()
    return True


# ------------------------------------------------------------------------------------------------ CFP_EEL

def held_elbereth(castle):
    """CFP_EEL: held by a giant eel (castle_logic saw 'swings itself around you' / 'cannot escape') while on foot:
    write Elbereth instead of hitting back. The next touch of a holder in the water drowns us (mhitu.c AD_WRAP); a
    scared holder lets go (monmove.c distfleeck -> monflee -> release_hero) and doesn't attack, while one blow kills
    a 5HD eel only sometimes. harness castle-c2 seed 3: the back door crashed open on the third kick, the eel at
    (57,07) swung itself round, two blows later 'drowned in a moat by a giant eel' one step from the trap door.
    True: engraved."""
    agent = castle.agent
    if castle.levitating() or agent.character.prop.blind or not agent.can_engrave():
        return False
    t = castle._tries
    if (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
        if 'You get released' in (agent.message or ''):
            return False
        # written and still held: the holder hasn't moved yet (a giant eel is speed 9, we are 12-18). Wait for its move --
        # scared, it lets go without attacking; a blow from this square erases the engraving (mon.c setmangry) and its
        # next touch drowns us (castle-c2 seed 10 under the first CFP_EEL: Elbereth written, the shark fled, the bot hit
        # the eel from the square and drowned the next turn)
        if t.get('cfp_held_wait', 0) >= 8:
            return False
        t['cfp_held_wait'] = t.get('cfp_held_wait', 0) + 1
        castle._set_state('cfp: held on Elbereth: waiting for the holder to let go')
        agent.search()
        return True
    if t.get('cfp_held_elbereth', 0) >= 12:
        return False
    t['cfp_held_elbereth'] = t.get('cfp_held_elbereth', 0) + 1
    castle._set_state('cfp: held by a sea monster on foot: Elbereth')
    agent.engrave('Elbereth')
    _log(castle, f'held at {castle._pos()}: Elbereth -> {agent.message[:80]!r}')
    return True


def door_sea_fight(castle):
    """CFP_EEL, on foot in front of the locked back door (57,08), nothing left but kicking: a sea monster in view next
    to us (the east eels start at (57,07)/(57,09), both next to this square) is fought before the next kick. Every
    kick wakes them (dokick.c wake_nearby) and wipes the Elbereth under us (u_wipe_engr), and a wrap there drowns us
    one step from the trap door -- castle_logic's _fight_adjacent skips the water. True: acted."""
    agent = castle.agent
    if castle.levitating():
        return False
    if (agent.inventory.engraving_below_me or '').lower() == 'elbereth':
        # (a sea monster next to our Elbereth is fleeing -- it keeps fleeing after the kick wipes the letters -- and
        # hitting it from here would erase it: mon.c setmangry 'You feel like a hypocrite', alignment -5)
        return False
    targets = _sea_targets(castle)
    if not targets:
        return False
    t = castle._tries
    if t.get('cfp_door_sea', 0) >= 60:
        return False
    t['cfp_door_sea'] = t.get('cfp_door_sea', 0) + 1
    _duel_hit(castle, targets[0], 'at the back door: hitting the sea monster at')
    return True


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


# ------------------------------------------------------------------------------------------------ CL_POTION_EARLY
#
# castle-lift: the kit's potions that may be levitation are quaffed where we stand at the first quiet moment on the
# castle's west side (see jf_config.CL_POTION_EARLY for the census behind it). castle_logic tries them only at
# TEST_SPOT (1,7), after the walk or dig through the west maze, and that walk runs at the very bottom of the preempt
# chain: in the fresh cand-g games 10 of 13 kits holding a real potion of levitation never quaffed it. A levitation
# potion found in the maze floats us at once, and rush_strategy's floating branch digs/floats straight onto the moat.

CL_QUIET = 3         # no monster glyph (not known peaceful) within this many squares before an unknown potion...
CL_QUIET_KNOWN = 4   # ...and a known potion of levitation (its turns start at once)
CL_HP = 0.5          # HP fraction for an unknown potion (a potion of sleeping/paralysis leaves us helpless 25-34 turns)
CL_HP_KNOWN = 0.7    # ...and for a known one (straight into the west channel's sharks)


def _hostile_glyphs_within(castle, r):
    """Monster glyphs (and remembered invisible ones) within r squares, not known peaceful, not in the water beyond reach
    (a sea monster two squares off can't bite a hero who isn't next to the water). Read from the glyphs, not
    get_visible_monsters: that drops monsters whose neighbourhood the BFS can't reach (the dark maze, minotaur lane)."""
    agent = castle.agent
    y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
    peaceful = agent.monster_tracker.peaceful_monster_mask
    h, w = agent.glyphs.shape
    out = []
    for y in range(max(0, y0 - r), min(h, y0 + r + 1)):
        for x in range(max(0, x0 - r), min(w, x0 + r + 1)):
            if (y, x) == (y0, x0):
                continue
            g = agent.glyphs[y, x]
            if not (nh.glyph_is_monster(g) or g == nh.GLYPH_INVISIBLE) or peaceful[y, x]:
                continue
            mx, my = _to_map(y, x)
            if _wet(castle, mx, my) and max(abs(y - y0), abs(x - x0)) > 1:
                continue
            out.append((y, x))
    return out


def _poly_drill_pending(castle):
    """castle_power's polymorph drill (CASTLE_POLY: a known wand of polymorph with charges, zapped at ourselves until a
    form that crosses water comes up, at most 12 zaps) still runs: the potions wait for it, as they do without
    CL_POTION_EARLY (castle_logic tests them after the drill). The castle-29 benchmark cl-fin-all cfpf-s4 quaffed its
    potion in a white unicorn form between two self-zaps: the forms that followed differed (a green mold, a panther
    fight, 2213 turns on the west side) and base's pass (an earth elemental through the walls, a couatl over the trap
    door) was lost; cfpj-s11 quaffed two potions as a 2-HP manes between zaps."""
    if not jf_config.CASTLE_POLY or castle._tries.get('pw_polyzaps', 0) >= 12:
        return False
    try:
        from . import castle_power
        if castle_power._poly_wand(castle.agent) is None:
            return False
        form = form_permonst(castle.agent)
        if form is not None and (castle_power.crosses(form) or amphibious(castle) or wallwalker(castle.agent)):
            return False   # (a form that crosses or walks the walls: the drill is over)
        return True
    except Exception:
        return False


def early_potion(castle):
    """CL_POTION_EARLY: the potion to quaff right here, or None (not quiet, hurt, next to water, no candidate)."""
    if not jf_config.CL_POTION_EARLY:
        return None
    agent = castle.agent
    bl = agent.blstats
    prop = agent.character.prop
    if castle.levitating() or prop.confusion or prop.stun:
        return None
    if (prop.hallu or prop.blind) and not jf_config.BREACH_NOWAIT:
        # (hallucination lasts 600-800 turns, a potion of blindness 250-450: the next potion only when those turns aren't
        # waited out anyway -- cl-t10-pe: 29 'hallucinogen-distorted' deaths among potion testers that never floated)
        return None
    t = castle._tries
    if t.get('cl_pot_n', 0) >= 30:
        return None
    if _poly_drill_pending(castle):
        return None
    from .power_route import known_cursed
    # (a known potion of polymorph stays power's; a known-cursed potion of levitation lifts for one turn: potion.c)
    pots = [p for p in potion_candidates(castle) if 'levitation' in _names(p) and not known_cursed(p)]
    if not pots:
        return None
    item = pots[0]
    known = item.is_unambiguous()
    if bl.hitpoints < (CL_HP_KNOWN if known else CL_HP) * bl.max_hitpoints:
        return None
    if castle._wet_around(*castle._pos()):
        return None   # next to the moat: a confused or stunned step walks into it, an eel wraps us while we sleep
    if _hostile_glyphs_within(castle, CL_QUIET_KNOWN if known else CL_QUIET):
        return None
    return item


def _early_potion_step(castle, item):
    """Elbereth under us (a potion of sleeping or paralysis leaves us helpless 25-34 turns; scared monsters don't
    melee), then quaff. True: acted."""
    agent = castle.agent
    t = castle._tries
    engraving = (agent.inventory.engraving_below_me or '').lower()
    # (blind -- BREACH_NOWAIT goes on testing -- a dust engraving can't be read back (engrave.c read_engr_at senses DUST
    # only with sight): one Elbereth, not eight, before each potion)
    elb_max = 1 if agent.character.prop.blind else 8
    if engraving != 'elbereth' and agent.can_engrave() and t.get('cl_pot_elb', 0) < elb_max:
        t['cl_pot_elb'] = t.get('cl_pot_elb', 0) + 1
        castle._set_state('cl: Elbereth before an early potion test')
        agent.engrave('Elbereth')
        return True
    t['cl_pot_n'] = t.get('cl_pot_n', 0) + 1
    t['cl_pot_elb'] = 0
    bl = agent.blstats
    _log(castle, f'early potion: quaffing {item.text!r} at {castle._pos()} hp {bl.hitpoints}/{bl.max_hitpoints} '
                 f'turn {bl.time}')
    castle._try('potion', item)
    if castle._floating():
        _log(castle, f'early potion: floating on {item.text!r} at {castle._pos()}: to the moat')
    return True


# ------------------------------------------------------------------------------------------------ CL_ROUTE
#
# castle-lift: a floating hero's way from the west maze onto the moat, planned over the maze's real structure instead of
# rush_strategy's straight rows-first dig to the fixed launch squares (-1,2)/(-1,14). mkmaze.c walkfrom carves cells at
# odd LEVEL coordinates (sp_lev.c spo_mazewalk forces the odd parity), i.e. castle map squares (even x, even y) for
# x in -6..-2: a cell is always open, a pillar (odd x, odd y) is always wall, a square between two cells is open about
# half the time (a spanning tree), and the boundary column x = -1 is wall except the courtyard's exit (-1,10). The old
# rows-first path from cl-h-pe1 cm-jf40-s5~2 (-5,8) dug up the pillar column x = -5: 7 digs, 37 turns from 'floating'
# to the first moat square (median 20 over the harness's floating potion kits) -- turns a potion's 10-149 of lift can't
# spare and a maze minotaur uses. Dijkstra over the maze squares with dig costs, to the moat square whose rest of the
# west channel is cheapest (water moves to the dry strip x 0.75 turns), sea monsters seen by an entry cost extra.

CL_DIG = 3.0          # turns a dwarf's pick takes through a maze wall (dig.c: effort += 2 x (10 + d5 + bonuses) > 100)
CL_WATER = 0.75       # turns per moat square afloat with intrinsic speed
CL_SEA = 8.0          # extra cost of an entry square with a sea monster on or next to it (seen lately)
CL_WET_DIG = 10.0     # extra cost of a dig made from a square beside the water (the bites stop the dig)
CL_MAZE_X = (-6, -1)  # the west maze's columns (level x 3..8)
_CL_WATER_DIST = None


def _cl_mouth(w):
    """Extra cost of entering the moat at w by how close it is to the courtyard's moat (row 5, column x 5, row 11): the
    west sharks home in on us (monmove.c m_move: gx,gy = mux,muy, always our square) and settle at the water square
    nearest us, which from the courtyard or the maze's middle rows is the channel mouth (0,5)/(1,5) or (0,11)/(1,11); and
    the column there is 4-5 squares longer. Harness crossings by entry (castle-lift cl-h-base2/pe1/rt1, pooled):
    (0,5) 2 of 38 passed, (1,5) 1/6, (0,4) 1/4, (0,13) 0/3 -- but (0,1) 11/18 and (0,15) 8/19."""
    x, y = w
    if x == 0 and y <= 16:
        far = min(y, 16 - y) if y <= 5 or y >= 11 else 99
        return {0: 0.0, 1: 0.0, 2: 3.0, 3: 8.0}.get(far, 15.0)
    return 15.0   # row 5, column x 5, row 11: the courtyard's own moat


def _cl_water_dist():
    """Moat square (west part) -> moves afloat to the first dry strip square (9,0) / (9,16)."""
    global _CL_WATER_DIST
    if _CL_WATER_DIST is None:
        from .castle_logic import MAP_H
        dist = {}
        todo = []
        for start in ((9, 0), (9, 16)):
            dist[start] = 0
            todo.append(start)
        i = 0
        while i < len(todo):
            x, y = todo[i]
            i += 1
            for dx, dy in DIRS8:
                n = (x + dx, y + dy)
                if n in dist or not (0 <= n[0] <= 8 and 0 <= n[1] < MAP_H) or map_char(*n) != '}':
                    continue
                dist[n] = dist[(x, y)] + 1
                todo.append(n)
        _CL_WATER_DIST = {p: d for p, d in dist.items() if map_char(*p) == '}'}
    return _CL_WATER_DIST


def _cl_kind(castle, mx, my):
    """'open' / 'wall' / 'unknown' / 'boulder' / None (off the land we plan over) for a west maze or west courtyard
    square, from the map memory and the maze's parity."""
    court = 0 <= mx <= 4 and 6 <= my <= 10
    if not court and not (CL_MAZE_X[0] <= mx <= CL_MAZE_X[1] and 0 <= my <= 16):
        return None
    agent = castle.agent
    level = agent.current_level()
    y, x = to_bot(mx, my)
    g = agent.glyphs[y, x]
    obj = level.objects[y, x]
    if g in G.BOULDER or obj in G.BOULDER:
        return 'boulder'
    if court or level.walkable[y, x]:
        return 'open'
    if (mx % 2 == 0 and my % 2 == 0 and mx <= -2) or (mx, my) == (-1, 10):
        return 'open'      # a maze cell (carved by walkfrom), or the courtyard's exit (castle.des MAZEWALK:(00,10),west)
    if (mx % 2 and my % 2) or (mx == -1 and my != 10):
        return 'wall'      # a pillar, or the boundary column beside the map (only (-1,10) is carved)
    if g in G.WALL or obj in G.WALL:
        return 'wall'
    return 'unknown'


def _cl_rock(castle, mx, my):
    """Wall or rock for the diagonal-squeeze rule (maze walls and unknown squares count; floor and water don't)."""
    if mx >= 0:
        return map_char(mx, my) not in '.}'
    return _cl_kind(castle, mx, my) in ('wall', 'unknown', None)


def _cl_threat(castle):
    """A land monster that ignores Elbereth (minotaur, @, ...) within 5 squares, from the glyphs (the dark maze hides
    them from get_visible_monsters' BFS filter)."""
    agent = castle.agent
    bl = agent.blstats
    for m in agent.get_visible_monsters():
        if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 5 and castle.dive._ignores_elbereth(m[3]) and \
                not _wet(castle, *_to_map(m[1], m[2])):
            return True
    try:
        from . import mino_guard
        mask = mino_guard._MINO_GLYPHS
        y0, x0 = int(bl.y), int(bl.x)
        g = agent.glyphs[max(0, y0 - 5):y0 + 6, max(0, x0 - 5):x0 + 6]
        import numpy as np
        return bool(np.isin(g, list(mask)).any())
    except Exception:
        return False


def _cl_can_dig(castle):
    """A digging tool we can swing. A form without hands can't (cl-pe1-c29 jf43 s14, castle 29: a golden naga floated on
    its potion of levitation one square from the launch square it had to dig, 'rush stuck', until the potion ran out):
    then the route walks the maze's corridors to the open courtyard exit (-1,10) and floats onto (0,11) from there."""
    if castle.dive.digging_tool() is None:
        return False
    form = form_permonst(castle.agent)
    if form is not None and (castle._tries.get('cfp_nodig') == form.mname or form.mflags1 & M1_NOHANDS):
        return False
    return True


def _cl_wet_adj(castle, p):
    """A moat square that is still water next to map square p: sea monsters bite anything standing on p."""
    return any(_wet(castle, p[0] + dx, p[1] + dy) for dx, dy in DIRS8)


# a known trap on a maze square: a web holds a floating hero and takes the lift away while it holds (trap.c
# float_vs_flight: 'being trapped on the ground ... overrides floating' -- BLevitation, the status line drops Lev);
# cl-fin-all jf48-s12 (castle 29) floated into a web at (-4,2) on its way to the moat, castle_logic saw no lift and
# gave the passage up, BREACH_RESUME brought it back 15-25 turns later, and cl_route led it into the same web six times
# until the potion ran out. Pits, holes, trap doors, bear traps, squeaky boards and land mines don't touch a levitator.
_CL_WEB = frozenset({SS.S_web})
_CL_TRAP_HURTS = frozenset({SS.S_sleeping_gas_trap, SS.S_polymorph_trap, SS.S_magic_trap, SS.S_anti_magic_trap,
                            SS.S_fire_trap, SS.S_level_teleporter, SS.S_teleportation_trap, SS.S_rust_trap,
                            SS.S_arrow_trap, SS.S_dart_trap, SS.S_falling_rock_trap, SS.S_statue_trap,
                            SS.S_rolling_boulder_trap, SS.S_magic_portal})
CL_TRAP_WEB = 25.0    # route cost of a known web (a way round it through the maze is nearly always cheaper)
CL_TRAP = 5.0         # ...and of the other traps a levitator sets off


def _cl_trap_cost(castle, mx, my):
    agent = castle.agent
    y, x = to_bot(mx, my)
    seen = (agent.glyphs[y, x], agent.current_level().objects[y, x])
    if any(g in _CL_WEB for g in seen):
        return CL_TRAP_WEB
    if any(g in _CL_TRAP_HURTS for g in seen):
        return CL_TRAP
    return 0.0


WEB_IN = ('into a spider web', 'into your spider web', 'You are stuck to the web')
WEB_OUT = ('You disentangle yourself', 'You tear through', 'cuts through the web', 'You are no longer stuck',
           'You float gently to the', 'You float down')


def _cl_webbed(castle):
    """CL_ROUTE: held in a web on the way to the moat (the last messages), the status line's Lev gone while it holds."""
    agent = castle.agent
    if castle.levitating():
        return False
    held = False
    for m in agent._message_history[-6:] + [agent.message or '']:
        if any(s in m for s in WEB_IN):
            held = True
        if any(s in m for s in WEB_OUT):
            held = False
    return held and castle._tries.get('cl_web_tries', 0) < 12


def _cl_web_pull(castle):
    """One move attempt out of the web (trap.c trapmove: each attempt takes one off u.utrap, which a St 18+ hero's web
    starts at 1, and the attempt itself doesn't move us), toward a dry square: toward the route's next square when it is
    dry land (never the water: an attempt made after the web already let go is a real step, on foot)."""
    pos = tuple(int(v) for v in castle._pos())
    plan = cl_route(castle)
    cands = []
    if plan is not None and _cl_kind(castle, *plan[0][0]) == 'open' and not _wet(castle, *plan[0][0]):
        cands.append(plan[0][0])
    for dx, dy in DIRS8:
        n = (pos[0] + dx, pos[1] + dy)
        if _cl_kind(castle, *n) == 'open' and not _wet(castle, *n) and not castle._monster_at(*n) and \
                not (dx and dy and _cl_rock(castle, pos[0] + dx, pos[1]) and _cl_rock(castle, pos[0], pos[1] + dy)):
            cands.append(n)
    if not cands:
        castle.agent.search()
        return
    castle._set_state(f'cl route: pulling out of the web toward {cands[0]}')
    castle._step_to(*cands[0])


def cl_route(castle, goals=None):
    """Cheapest (turns) plan from our square to a moat entry: (path [squares, the last one the moat square], half,
    cost), or None (not in the west maze, or no way to the water). goals: only these moat squares (a kept target)."""
    import heapq
    pos = tuple(int(v) for v in castle._pos())
    if _cl_kind(castle, *pos) is None:
        return None
    dig = _cl_can_dig(castle)
    wdist = _cl_water_dist()
    agent = castle.agent
    now = agent.blstats.time
    seen = castle._tries.setdefault('cl_sea_seen', {})
    blocked = castle._tries.setdefault('cl_blocked', {})
    # CL_FLEE: an Elbereth-ignorer on land close by (a minotaur: 3d10/3d10/2d8 a turn, it can't follow into the water) --
    # the nearest water wins over the far channel entry (a mouth entry's waiting shark is the lesser evil)
    flee = jf_config.CL_FLEE and castle._floating() and _cl_threat(castle)
    mouth_w = 0.2 if flee else 1.0

    def step_cost(p, n):
        k = _cl_kind(castle, *n)
        if k is None:
            return None
        c = 1.0
        digging = False
        if k == 'wall':
            if not dig:
                return None
            c += CL_DIG
            digging = True
        elif k == 'boulder':
            if not dig:
                return None
            c += CL_DIG
            digging = True
        elif k == 'unknown':
            if not dig:
                c += 2.0
            else:
                c += 0.5 * CL_DIG
                digging = True
        if castle._monster_at(*n):
            c += 4.0
        c += _cl_trap_cost(castle, *n)
        if _cl_wet_adj(castle, n):
            c += 1.0   # a square on the moat's edge: eels and sharks bite from the water
        if digging and _cl_wet_adj(castle, p):
            # digging from a square beside the water: each bite stops the dig (cl-t10-rt cg-jf55-s6~2 dug (-1,14) from
            # (-1,13) through four 'The shark bites!  You stop digging.' and died there)
            c += CL_WET_DIG
        return c

    def entry_cost(w):
        c = 1.0 + CL_WATER * wdist[w] * (0.3 if flee else 1.0) + _cl_mouth(w) * mouth_w
        for p, t in seen.items():
            if now - t <= 30 and max(abs(p[0] - w[0]), abs(p[1] - w[1])) <= 1:
                c += CL_SEA
        if blocked.get(w, 0) >= 2:
            c += 20.0
        return c

    best = None
    dist = {pos: 0.0}
    prev = {}
    heap = [(0.0, pos)]
    while heap:
        d, p = heapq.heappop(heap)
        if d > dist.get(p, 1e9):
            continue
        if best is not None and d >= best[0]:
            break
        for dx, dy in DIRS8:
            n = (p[0] + dx, p[1] + dy)
            if n in wdist and not castle._dry(*n):
                if goals is not None and n not in goals:
                    continue
                if dx and dy and _cl_rock(castle, p[0] + dx, p[1]) and _cl_rock(castle, p[0], p[1] + dy):
                    continue   # (a squeeze between two walls: hack.c test_move refuses it when heavily loaded)
                tot = d + entry_cost(n)
                if best is None or tot < best[0]:
                    best = (tot, p, n)
                continue
            if dx and dy:
                # diagonal steps only between known open squares (no digging diagonally, no squeezes)
                if _cl_kind(castle, *n) != 'open' or _cl_kind(castle, p[0] + dx, p[1]) != 'open' and \
                        _cl_kind(castle, p[0], p[1] + dy) != 'open':
                    continue
            c = step_cost(p, n)
            if c is None:
                continue
            nd = d + c
            if nd < dist.get(n, 1e9):
                dist[n] = nd
                prev[n] = p
                heapq.heappush(heap, (nd, n))
    if best is None:
        return None
    _, last, w = best
    path = [w, last]
    while path[-1] != pos:
        path.append(prev[path[-1]])
    path.reverse()
    half = 'north' if w[1] <= 8 else 'south'
    return path[1:], half, best[0]


def _cl_zap_dig(castle, n, path):
    """A known wand of digging with a charge to spare tunnels the straight run of walls ahead in one zap (zap.c zap_dig:
    through every diggable wall until its range, 8..25, runs out; the castle map itself is NON_DIGGABLE and stops it).
    True: zapped."""
    agent = castle.agent
    wand = next((i for i in castle._items() if castle._usable_wand(i, 'digging')), None)
    if wand is None:
        return False
    m = re.search(r'\(\d+:(\d+)\)', wand.text or '')
    if m and int(m.group(1)) < 2:
        return False   # (keep the last charge: the back door, LIFT_DOOR_RAYS/door_names)
    pos = castle._pos()
    d = (int(n[0] - pos[0]), int(n[1] - pos[1]))
    if d[0] and d[1]:
        return False
    walls = 0
    cur = (int(pos[0]), int(pos[1]))
    for sq in path:
        if (sq[0] - cur[0], sq[1] - cur[1]) != d:
            break
        if _cl_kind(castle, *sq) in ('wall', 'unknown'):
            walls += 1
        cur = sq
    if walls < 2 or castle._tries.get('cl_zapdig', 0) >= 3:
        return False
    castle._tries['cl_zapdig'] = castle._tries.get('cl_zapdig', 0) + 1
    y, x = to_bot(*n)
    direction = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
    castle._set_state(f'cl route: zapping {wand.text!r} {direction} through {walls} walls')
    agent.zap(wand, direction)
    _log(castle, f'cl route: zap dig {direction} from {pos}: {agent.message[:90]!r}')
    if 'Nothing happens' in agent.message or 'You wrest' in agent.message:
        agent.inventory.empty_wands.add(wand.text)
    agent.last_bfs_step = -1
    return True


def _cl_note_sea(castle):
    """Sea monsters seen on moat squares within 3 of us: square -> turn (cl_route's entry costs)."""
    agent = castle.agent
    seen = castle._tries.setdefault('cl_sea_seen', {})
    pos = castle._pos()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            p = (int(pos[0]) + dx, int(pos[1]) + dy)
            if map_char(*p) == '}' and not castle._dry(*p) and castle._monster_at(*p):
                seen[p] = agent.blstats.time


def cl_route_step(castle, on_foot=False, goals=None):
    """CL_ROUTE: one step, dig or zap along cl_route. True: acted; None: no plan (the caller's old way); 'launch' (on
    foot): the next step would be the water -- we stand on the launch square."""
    agent = castle.agent
    t = castle._tries
    pos = castle._pos()
    now = agent.blstats.time
    if form_permonst(agent) is not None:
        # a polymorph form: castle_power's (castle-poly) way, as without CL_ROUTE -- cl-best-old cfpf-s4's green mold
        # 'walked' in place until it starved, and cl-fix5 cfpf-s4 walked its 29-HP energy vortex into a panther in the
        # maze where base waited the form out and passed on the forms that followed
        return None
    # no progress: the same square for 8 route steps -> no route for 20 turns (the caller's old way, whose stuck count
    # hands over to castle_power's re-polymorph / castle_logic)
    key = tuple(int(v) for v in pos)
    if t.get('cl_same_pos') == key:
        t['cl_same_n'] = t.get('cl_same_n', 0) + 1
    else:
        t['cl_same_pos'] = key
        t['cl_same_n'] = 0
    if now < t.get('cl_route_off_until', -1):
        return None
    if t['cl_same_n'] >= 8:
        t['cl_same_n'] = 0
        t['cl_route_off_until'] = now + 20
        _log(castle, f'cl route: no progress at {key}: off for 20 turns')
        return None
    _cl_note_sea(castle)
    plan = cl_route(castle, goals=goals)
    if plan is None and goals is not None:
        plan = cl_route(castle)
    if plan is None:
        return None
    path, half, cost = plan
    n = path[0]
    if t.get('cl_route_logged') != path[-1]:
        t['cl_route_logged'] = path[-1]
        _log(castle, f'cl route: from {pos} to the moat at {path[-1]} ({half}), ~{cost:.0f} turns: {path}')
    if map_char(*n) == '}' and on_foot:
        return 'launch'
    if map_char(*n) == '}':
        castle._half = half
        if castle._monster_at(*n):
            blocked = t.setdefault('cl_blocked', {})
            blocked[n] = blocked.get(n, 0) + 1
        castle._set_state(f'cl route: onto the moat at {n}')
        castle._step_to(*n)
        return True
    if castle._monster_at(*n):
        if not t.get('wielded'):
            t['wielded'] = 1
            if _swap_for_fight(castle):
                return True
        castle._set_state(f'cl route: attacking what blocks {n}')
        castle._step_to(*n)
        return True
    k = _cl_kind(castle, *n)
    y, x = to_bot(*n)
    g = agent.glyphs[y, x]
    level = agent.current_level()
    if k in ('wall', 'boulder') or (k == 'unknown' and (g in G.WALL or g in G.STONE)):
        if not _cl_can_dig(castle):
            # no pick we can swing: remember the wall, the next plan goes round it
            level.walkable[y, x] = False
            level.objects[y, x] = SS.S_vwall
            t[('cl_wall', n)] = 1
            return True
        if k != 'boulder' and _cl_zap_dig(castle, n, path):
            return True
        key = ('cl_dig', n)
        if t.get(key, 0) >= 5:
            level.walkable[y, x] = False
            return None
        t[key] = t.get(key, 0) + 1
        castle._set_state(f'cl route: digging {n}')
        if not _dig(castle, n):
            return True   # (_dig noted a form that can't hold the pick: cfp_nodig -> the next plan walks)
        return True
    castle._set_state(f'cl route: to {n}')
    before = castle._pos()
    castle._step_to(*n)
    if castle._pos() == before and not castle._monster_at(*n):
        # the move didn't happen (an unseen wall): treat it as wall from now on
        key = ('cl_nomove', n)
        t[key] = t.get(key, 0) + 1
        if t[key] >= 2:
            level.walkable[y, x] = False
            level.objects[y, x] = SS.S_vwall
    return True


# ------------------------------------------------------------------------------------------------ CL_LAUNCH
#
# castle-lift: the kit's lifts are tried ON the launch square, not where we landed. On foot the hero walks/digs
# cl_route's way to the land square next to the chosen far channel entry, writes Elbereth there and only then tries
# the unknown rings and the potions that may be levitation; a lift steps straight onto the water. Why (the harness
# lift suite, 45 real kits x 10 salts, cl-t10-*): (1) crossings that reach the moat within 30 turns of the landing
# pass far more often (far entries 59/157 vs 12/82 after 60 turns: the throne room's xorns reach the west towers'
# walls after ~30 turns), and a potion found in the maze burned ~15 turns of its 10-149 afloat on the way to the water;
# (2) the channel's shark waits at the water square next to the hero, and 26 floating heroes died on the launch square
# fighting it (cl-t10-rt) -- a sea monster next to our Elbereth is scared (monmove.c distfleeck: monflee rnd(10), no
# attack) and keeps fleeing for a few turns after we step off it; (3) on foot we can write Elbereth for the tests
# (a potion of sleeping or paralysis: 25-34 helpless turns).

CL_LAUNCH_KEEP = 8.0   # a new plan replaces the kept launch target only when it is this much cheaper


def _cl_lift_potions(castle):
    from .power_route import known_cursed
    return [p for p in potion_candidates(castle) if 'levitation' in _names(p) and not known_cursed(p)]


def _cl_launch_target(castle):
    """(L, W, path) -- the launch square L (land, next to the entry W) and cl_route's path to W from here, kept from step
    to step unless a new plan is CL_LAUNCH_KEEP cheaper or the kept entry is blocked. None: no way to the water."""
    t = castle._tries
    plan = cl_route(castle)
    if plan is None:
        return None
    path, half, cost = plan
    cur = t.get('cl_launch')
    if cur is not None and cur != path[-1] and t.setdefault('cl_blocked', {}).get(cur, 0) < 2:
        keep = cl_route(castle, goals={cur})
        if keep is not None and keep[2] <= cost + CL_LAUNCH_KEEP:
            path, half, cost = keep
    w = path[-1]
    t['cl_launch'] = w
    land = path[-2] if len(path) >= 2 else tuple(int(v) for v in castle._pos())
    return land, w, path


def cl_launch_plan(castle):
    """CL_LAUNCH, not floating (see the section comment): ('cl_walk', goals), ('cl_elbereth',), ('lift', kind, item),
    ('cl_potion', item), ('cl_wait',) or None (no lift left to try here, a fight next to us, no way to the water)."""
    agent = castle.agent
    bl = agent.blstats
    t = castle._tries
    cands = lift_candidates(castle)
    pots = _cl_lift_potions(castle)
    if not cands and not pots:
        return None
    if t.get('cl_launch_off') or t.get('cl_launch_steps', 0) > 400:
        return None
    tgt = _cl_launch_target(castle)
    if tgt is None:
        return None
    land, w, path = tgt
    pos = tuple(int(v) for v in castle._pos())
    prop = agent.character.prop
    ignorers = [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 3 and castle.dive._ignores_elbereth(m[3]) and
                not _wet(castle, *_to_map(m[1], m[2]))]
    if pos != land:
        if _land_hostiles_within(castle, 1) or ignorers or prop.confusion or prop.stun:
            return None   # a fight: the usual layers (fight2, the scare hold, MINO_GUARD)
        if bl.hitpoints < 0.4 * bl.max_hitpoints and t.get('cl_launch_rest', 0) < 150:
            return ('cl_wait',)
        return ('cl_walk', {w})
    # on the launch square
    engraving = (agent.inventory.engraving_below_me or '').lower()
    if engraving != 'elbereth' and agent.can_engrave() and t.get('cl_launch_elb', 0) < 12:
        return ('cl_elbereth',)
    if ignorers or prop.confusion or prop.stun:
        return None
    if cands:
        return ('lift',) + cands[0]
    if bl.hitpoints < 0.4 * bl.max_hitpoints and t.get('cl_launch_rest', 0) < 150:
        return ('cl_wait',)
    if prop.hallu and not jf_config.BREACH_NOWAIT:
        return ('cl_wait',) if t.get('cl_launch_rest', 0) < 150 else None
    return ('cl_potion', pots[0])


def _land_hostiles_within(castle, r):
    agent = castle.agent
    bl = agent.blstats
    return [m for m in agent.get_visible_monsters()
            if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= r and not _wet(castle, *_to_map(m[1], m[2]))]


def _adjacent_land_hostiles(castle):
    return _land_hostiles_within(castle, 1)


def _swap_for_fight(castle):
    """The 'real weapon' for a fight in the west maze, once after each dig (_dig clears 'wielded'): True when the wield
    took the move. CASTLE_PICK_MELEE: not while the pick-axe/mattock in hand is about as good -- DIG_TOOL_MELEE's rule
    (agent._keep_digging_tool_wielded: the best weapon must beat it by DIG_TOOL_MELEE_MARGIN in expected damage). Afloat on
    a potion the swap is a turn and the next dig's apply re-wields the pick (another), out of 10-149 turns of lift."""
    agent = castle.agent
    if jf_config.CASTLE_PICK_MELEE and agent._keep_digging_tool_wielded():
        return False
    return agent.wield_best_melee_weapon()


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
        if _swap_for_fight(castle):
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


def _xorn_path(castle, start, goals=None, blocked=(), moat_cost=20, moat_side_cost=4):
    """Cheapest way for a wall-walker from start to a trap door (40..55,08) -- or to `goals` (CASTLE_TREASURY: the
    tower cells), never through `blocked` (the trap doors on the way there): through walls, doors and rock, around the
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
            c += moat_cost
        elif any(map_char(mx + dx, my + dy) == '}' for dx, dy in DIRS8):
            # beside the moat: its sharks and eels bite into the wall (harness cfp-xorn1 lost 36 of 44 form HP along
            # row 2, next to the moat's row 1; the corridor on row 3 has walls on both sides)
            c += moat_side_cost
        if p in mons:
            c += 60
        elif any((mx + dx, my + dy) in mons for dx, dy in DIRS8):
            c += 8
        return c

    goals = set(TRAPDOORS if goals is None else goals)
    blocked = set(blocked)
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
            if n in blocked:
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
            # CASTLE_TREASURY (castle_treasury.py, from vlomshakov f84a81b): first through the walls to the four tower
            # cells, whose chest holds the castle's wand of wishing (castle.des) -- a bounded detour that hands back
            # to this walk on success or failure
            if jf_config.CASTLE_TREASURY:
                from . import castle_treasury
                if castle_treasury.step(castle):
                    continue
            pos = castle._pos()
            if jf_config.CASTLE_TREASURY and getattr(castle, '_treasury', None) is not None:
                # after the tower detour: the way on from an east tower runs along the corner moat (real arm jf16 s0:
                # the wand in hand, 38 -> 15 form HP at (55,1) beside it in 2 turns); inside the castle instead
                path = _xorn_path(castle, pos, moat_cost=60, moat_side_cost=20)
            else:
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


def eyeless_form(agent):
    """Our polymorph form has no eyes (M1_NOEYES: blind while it lasts)."""
    form = form_permonst(agent)
    return form is not None and bool(form.mflags1 & 0x1000)


def _can_pray(castle):
    """A prayer would likely work now (the emergency layer prays away a cursed amulet of strangulation)."""
    agent = castle.agent
    try:
        return not agent.prayer_failed and agent.is_safe_to_pray(500)
    except Exception:
        return False


def door_wands():
    """Wands that open the locked back door when zapped at it: striking / digging / opening (beams), and with
    LIFT_DOOR_RAYS fire, lightning and cold rays too -- zap.c zap_over_floor: a closed door is consumed in flames /
    splinters / freezes and shatters, and the ray stops there (rangemod -1000: no bounce back). cmp-main jf41-s9 floated
    a potion's levitation right round a Dlvl-27 castle to the locked back door with a blessed wand of lightning (0:6) in
    the pack, 'waiting for the levitation to end' before it could kick; a minotaur from the east maze killed it 3 turns
    later."""
    names = ('striking', 'digging', 'opening')
    if jf_config.LIFT_DOOR_RAYS:
        names += ('fire', 'lightning', 'cold')
    return names


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
    names = door_wands()
    if castle.levitating() and castle._timed_levitation():
        names = names[:3]   # (rays: only once the potion's levitation is over -- see castle_logic._door_step)
    for name in names:
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


# ------------------------------------------------------------------------------------------------ LIFT_PLUNGE

MZ_HUGE = 4


def on_castle(castle):
    agent = castle.agent
    return castle.castle_key is not None and agent.current_level().key() == castle.castle_key


def plunge_strategy(dive):
    """LIFT_PLUNGE (lift-ready): standing on one of the castle's trap doors (40..55,08) and not levitating, press '>'.
    do.c dodown: on a trap door we have seen (uescaped_shaft) '>' is dotrap(TOOKPLUNGE), and trap.c fall_through on
    the stronghold goes to find_hell() -- the Valley. Only levitation, being held, or a huge form refuse it (a flyer or
    a clinger falls with TOOKPLUNGE). The real castle-29 game amd cfpf-s4 walked an earth-elemental form through the
    castle's walls onto (40,08): 'A trap door opens up under you!  You don't fit through.' (MZ_HUGE); the xorns there
    killed the form and it stood on the trap door in its own form with 45 HP while CFP_XORN looked for 'a way to a
    trap door' ('no way to a trap door from (40, 8)') until it died. A huge form zaps the wand of polymorph again (a
    smaller form falls); without one it waits on the trap door for the form to end. Top of the preempt chain: the
    Valley is banked progress (one level deeper) whatever our HP."""
    def f():
        castle = dive.castle
        agent = dive.agent
        if not jf_config.LIFT_PLUNGE or not on_castle(castle) or castle._pos() not in TRAPDOORS or \
                castle.levitating():
            yield False
            return
        t = castle._tries
        pos = castle._pos()
        key = ('lift_plunge', pos)
        now = agent.blstats.time
        if t.get(key, 0) >= 12 or t.get('lift_plunge_wait', -1) == now:
            yield False
            return
        yield True
        form = form_permonst(agent)
        if form is not None and form.msize >= MZ_HUGE and t.get(('lift_plunge_huge', pos)):
            # 'You don't fit through.' already: a smaller form, or wait for this one to end
            from . import castle_power
            wand = castle_power._poly_wand(agent) if jf_config.CASTLE_POLY else None
            if wand is not None and t.get('lift_plunge_zaps', 0) < 4:
                t['lift_plunge_zaps'] = t.get('lift_plunge_zaps', 0) + 1
                _log(castle, f'plunge: a {form.mname} does not fit through {pos}: zapping {wand.text!r} at ourselves')
                castle_power._zap_self(castle, wand)
                return
            castle._set_state(f'plunge: waiting on the trap door {pos} for the {form.mname} form to end')
            t['lift_plunge_wait'] = now
            agent.search()
            return
        t[key] = t.get(key, 0) + 1
        castle._set_state(f'plunge: on the trap door {pos}')
        agent.direction('>')
        msg = agent.message or ''
        _log(castle, f"plunge: '>' on the trap door {pos}: {msg[:120]!r}")
        if "don't fit through" in msg:
            t[('lift_plunge_huge', pos)] = 1
        elif "can't go down here" in msg:
            t[key] = 99   # not a trap door we have seen ('>' refuses, no turn): the crossing logic steps off and on

    return Strategy(f)


# ------------------------------------------------------------------------------------------------ LIFT_COLD
#
# The cold route, shortened. castle_logic's cold route starts at the courtyard corner (0,6), climbs the west column and
# goes all the way round: 28 moat squares. A cold ray freezes 2-4 of them (zap.c zap_over_floor: every frozen pool
# costs the ray 3 more of its rn1(7,7) range; the whole castle map is unlit, so only the square next to us shows), and
# a wand has 4-8 charges less the engrave test -- no real kit ever froze its way round (cfpi-s1: stranded at (3,0)
# when the wand ran out; harness cold1: 0/36). The short way: dig to the maze square (-1,R) beside the end of moat row R
# (0 north, 16 south; the maze outside the map is diggable, castle.des NON_DIGGABLE covers the map only), freeze row R
# eastward in straight rays from the ice front (9 squares to the dry strip), walk the strip, freeze (54..62,R)
# (9 squares), and dig east into the east maze at (63,R): it joins the east courtyard (MAZEWALK (62,06); its walls are
# diggable too), so the east column (62,1..5) is never frozen. 18 moat squares, ~6 rays.

COLD_ROWS = {'north': 0, 'south': 16}
EAST_MAZE_X = 63


def _cold_half(castle):
    t = castle._tries
    half = t.get('cold_half')
    if half is None:
        half = 'north' if castle._pos()[1] <= 8 else 'south'
        t['cold_half'] = half
    return half


def _iced(castle, mx, my):
    """A moat square we can walk on: known ice, or an object shown on it (objects sink in water, so a corpse or an item
    on a moat square is lying on ice: castle_logic._dry missed the square where our ray killed an eel -- cfpi-s1 zapped
    the rest of its wand at (3,0) with the ice (4,0) under the eel's corpse)."""
    if map_char(mx, my) != '}':
        return map_char(mx, my) == '.'
    if castle._dry(mx, my) or (mx, my) in castle._tries.get('lift_cold_ice', ()):
        return True
    y, x = to_bot(mx, my)
    g = castle.agent.glyphs[y, x]
    return g in G.OBJECTS or g in G.BODIES


def _cold_zap(castle, cold, n):
    """Freeze toward the moat square n (a straight ray from here)."""
    agent = castle.agent
    y, x = to_bot(*n)
    d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
    t = castle._tries
    t['lift_cold_zaps'] = t.get('lift_cold_zaps', 0) + 1
    castle._set_state(f'lift cold: freezing {d} toward {n}')
    if cold.category == nh.WAND_CLASS:
        agent.zap(cold, d)
        if 'Nothing happens' in agent.message or 'You wrest' in agent.message:
            agent.inventory.empty_wands.add(cold.text)
    else:
        with agent.atom_operation():
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(cold))
            if 'In what direction?' in agent.single_message:
                agent.direction(d)
    agent.inventory.items.update(force=True)
    msg = agent.message or ''
    if 'Nothing happens' not in msg and 'You wrest' not in msg:
        # the first moat square a ray crosses always freezes (its range 7..13 covers the 4 that costs): remember it --
        # an eel or a corpse on it hides the ice glyph
        t.setdefault('lift_cold_ice', set()).add(tuple(n))
    _log(castle, f'lift cold: ray {d} from {castle._pos()}: {msg[:100]!r}')


def _cold_blocked(castle, n):
    """Something on the square n: attack it (a sea monster in the water ahead gets the ray instead)."""
    if not castle._monster_at(*n):
        return False
    agent = castle.agent
    if map_char(*n) == '}' and not _iced(castle, *n):
        return False   # in the water: the freezing ray hits it anyway
    if not castle._tries.get('wielded'):
        castle._tries['wielded'] = 1
        if agent.wield_best_melee_weapon():
            return True
    y, x = to_bot(*n)
    castle._set_state(f'lift cold: attacking what blocks {n}')
    with agent.atom_operation():
        agent.step(A.Command.FIGHT)
        agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
    return True


def _engrave_elbereth_here(castle):
    """Elbereth under us, on the dust or on the ice ('What do you want to write in the frost here?' -- agent.engrave
    only accepts the dust prompt and would mark an ice square unengravable). Sea monsters and xorns respect it (monmove.c
    onscary excludes only @, minotaurs, shopkeepers/guards/priests and the Riders). True: written."""
    agent = castle.agent
    ok = [False]

    def gen():
        if 'What do you want to write with?' not in agent.single_message:
            yield A.Command.ESC
            return
        yield '-'
        if 'Do you want to add to the current engraving?' in agent.single_message:
            yield 'n'
        while agent._observation['misc'][2]:
            yield ' '
        msg = agent.single_message or ''
        if 'What do you want to write in the dust here?' not in msg and \
                'What do you want to write in the frost here?' not in msg:
            yield A.Command.ESC
            return
        yield from 'Elbereth'
        yield '\r'
        ok[0] = True

    with agent.atom_operation():
        agent.step(A.Command.ENGRAVE, gen())
        agent.inventory.get_items_below_me()
    return ok[0]


def _cold_guard(castle, pos):
    """On the cold row: Elbereth under us before acting while something bites us (a shark or an eel from the water, a
    xorn from the tower wall next to the moat's end: harness cold2 jf42-s3 died to xorns on row 16 x 1-2, cfpe-s0~3 to a
    shark), once per square; and on a dry strip square below half HP, rest on it to 80% (the ice ahead isn't made yet,
    the ice behind doesn't matter). True: acted."""
    agent = castle.agent
    t = castle._tries
    bl = agent.blstats
    near = [m for m in agent.get_visible_monsters() if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 1]
    engraving = (agent.inventory.engraving_below_me or '').lower()
    key = ('lift_cold_elb', tuple(pos))
    if near and engraving != 'elbereth' and t.get(key, 0) < 2 and agent.can_engrave() and \
            not agent.character.prop.blind:
        t[key] = t.get(key, 0) + 1
        castle._set_state(f'lift cold: Elbereth at {pos} ({[getattr(m[3], "mname", "?") for m in near]})')
        _engrave_elbereth_here(castle)
        return True
    # rest spots: a dry strip square, or the launch square before the first ray (cfpe-s0~3 set off at 29/98 HP) -- never
    # beside something Elbereth doesn't stop (a minotaur walks on the ice)
    ignorer = any(castle.dive._ignores_elbereth(m[3]) for m in near)
    restable = (map_char(*pos) == '.' or (pos[0] == -1 and not t.get('lift_cold_zaps'))) and not ignorer
    if restable and (bl.hitpoints < 0.55 * bl.max_hitpoints or t.get('lift_cold_resting')):
        if bl.hitpoints >= 0.85 * bl.max_hitpoints or t.get('lift_cold_rests', 0) >= 300:
            t.pop('lift_cold_resting', None)
            return False
        t['lift_cold_resting'] = 1
        t['lift_cold_rests'] = t.get('lift_cold_rests', 0) + 1
        if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind and \
                t.get(('lift_cold_elb', tuple(pos)), 0) < 4:
            t[('lift_cold_elb', tuple(pos))] = t.get(('lift_cold_elb', tuple(pos)), 0) + 1
            castle._set_state(f'lift cold: Elbereth to rest on the strip at {pos}')
            _engrave_elbereth_here(castle)
            return True
        castle._set_state(f'lift cold: resting at {pos}')
        agent.search(3)
        return True
    t.pop('lift_cold_resting', None)
    return False


def cold_pending(castle):
    """LIFT_COLD takes the castle: a known cold source, no lasting lift to try first (CFP_RUSH), not floating, and we are
    on the west side, on the cold row, or in the east maze on the way to the courtyard."""
    if not jf_config.LIFT_COLD or not castle.active() or castle._floating():
        return False
    t = castle._tries
    if t.get('lift_cold_off'):
        return False
    if castle._cold_source() is None:
        return False
    if jf_config.CFP_RUSH and lift_candidates(castle):
        return False
    mx, my = castle._pos()
    row = COLD_ROWS[_cold_half(castle)]
    if my == row and -1 <= mx <= 62:
        return True
    if not ((mx, my) not in OUTSIDE or (mx, my) in WEST_COURTYARD):
        return False
    # in the mazes or the west courtyard on foot: a land hostile within 2 (or low HP, confusion) goes to the usual
    # layers first (Elbereth, fight2, the rest), as for CFP_RUSH's potion walk
    return not _on_foot_blocked(castle)


def cold_strategy(dive):
    """LIFT_COLD (see the section comment): the short cold route, above castle_logic's crossing (which would walk its own
    long route from the moment we stand on the ring)."""
    def one_step(castle):
        agent = castle.agent
        t = castle._tries
        pos = castle._pos()
        mx, my = int(pos[0]), int(pos[1])
        half = _cold_half(castle)
        row = COLD_ROWS[half]
        launch = (-1, row)
        court_row = 6 if half == 'north' else 10
        if mx >= EAST_MAZE_X:
            # the east maze: down (up) the first maze column to the courtyard's row, then west into the courtyard
            if my != court_row:
                goal = (EAST_MAZE_X, court_row)
            else:
                goal = (62, court_row)
            if (mx, my) == (EAST_MAZE_X, court_row):
                castle._set_state('lift cold: into the east courtyard')
                castle._step_to(62, court_row)
                return True
            r = _toward(castle, goal, None)
            return r is not None
        if my != row or mx < -1 or (mx, my) in WEST_COURTYARD:
            # west side: dig straight to the launch square beside the end of the cold row (from the courtyard west into
            # the maze first: its north/south edge is the moat)
            goal = (-1, my) if mx >= 0 else launch
            r = _toward(castle, goal, None)
            if r is None:
                t['lift_cold_stuck'] = t.get('lift_cold_stuck', 0) + 1
                if t['lift_cold_stuck'] >= 3:
                    t['lift_cold_off'] = 1
                    _log(castle, f'lift cold: no way to {launch} from {pos}: back to the old cold route')
            return r is not None
        # on the cold row (the launch square, the ice, the strip): east
        if _cold_guard(castle, (mx, my)):
            return True
        if mx >= 62:
            n = (EAST_MAZE_X, row)
            castle._set_state('lift cold: digging into the east maze')
            if castle._monster_at(*n):
                return _cold_blocked(castle, n) or _toward(castle, n, None) is not None
            return _toward(castle, n, None) is not None
        n = (mx + 1, row)
        if _cold_blocked(castle, n):
            return True
        if _iced(castle, *n):
            castle._set_state(f'lift cold: walking east along row {row}')
            castle._step_to(*n)
            return True
        cold = castle._cold_source()
        key = ('lift_cold_at', pos)
        if cold is None or t.get(key, 0) >= 2:
            # the wand is empty (or two rays from here froze nothing we can see): castle_logic's stranded logic tries
            # the kit's potions from here, else gives up
            t['lift_cold_off'] = 1
            _log(castle, f'lift cold: stranded at {pos} (cold source {cold.text if cold else None!r}, '
                         f'{t.get("lift_cold_zaps", 0)} rays)')
            return False
        t[key] = t.get(key, 0) + 1
        _cold_zap(castle, cold, n)
        return True

    def f():
        castle = dive.castle
        agent = dive.agent
        if not cold_pending(castle):
            yield False
            return
        yield True
        if castle._tries.get('lift_cold_logged') is None:
            castle._tries['lift_cold_logged'] = 1
            _log(castle, f'lift cold: short route via the {_cold_half(castle)} row from {castle._pos()} with '
                         f'{castle._cold_source().text!r}')
        steps = 0
        last = None
        same = 0
        while steps < 400 and cold_pending(castle):
            steps += 1
            before = agent.step_count
            try:
                acted = one_step(castle)
            except (AgentPanic, AgentChangeStrategy, AgentFinished):
                raise
            except Exception as e:   # a bug here must not become a panic loop: the old routes take over
                castle._tries['lift_cold_off'] = 1
                _log(castle, f'lift cold: disabled after {e!r}')
                return
            if not acted:
                return
            if agent.step_count == before:
                agent.search()
            pos = castle._pos()
            # (a rest stop on the strip or the launch square stands still on purpose)
            same = same + 1 if pos == last and not castle._tries.get('lift_cold_resting') else 0
            last = pos
            if same >= 30:
                castle._tries['lift_cold_off'] = 1
                _log(castle, f'lift cold: no progress at {pos}')
                return

    return Strategy(f)


# ------------------------------------------------------------------------------------------------ LIFT_KNOWN_RUSH
#
# A KNOWN lasting lift goes on the moment we arrive where the castle's arrivals land, before the recognition dig: the
# dig costs +3..+7 turns ('too hard' only after a pit) and leaves us in its pit (2-7 turns to climb out), and CFP_RUSH
# waits for it -- cmp-main jf16-s0 (Dlvl-29 castle) floated only at +11 on its third ring, the master lich's summons
# killed it at +15. Levitation can't dig, so the castle is confirmed by ear instead (castle-first-pass CFP_SENSE: the
# castle's soldiers open and break doors -- a door sound within 3 turns of 101 of 101 castle arrivals, 1 on 185 maze
# levels below Medusa, which have no doors); silence for LIFT_LISTEN turns and the lift comes off again and the dive
# digs as before. Water walking boots and an amulet of magical breathing don't stop a dig: they just go on first.

DOOR_SOUNDS = ('You hear a door open', 'You hear a door crash open')
LIFT_LISTEN = 4


def _castle_likely(dive):
    """Where a fall or the stairs put us could be the castle's west landing: the Dungeons of Doom at depth >= 25, below a
    KNOWN Medusa (her levels have doors and land us at x <= 5 too), bot x <= 9 (castle.des levregion(1,0,10,20)), and
    this level isn't the recognised castle yet."""
    from .level import Level
    agent = dive.agent
    level = agent.current_level()
    bl = agent.blstats
    if level.dungeon_number != Level.DUNGEONS_OF_DOOM or bl.depth < 25 or bl.x > 9:
        return False
    medusa = dive.medusa_level
    if medusa is None or medusa == level.key() or medusa[1] >= level.level_number:
        return False
    return dive.castle.castle_key != level.key()


def _known_lasting_lift(castle):
    """(kind, item): a KNOWN lasting lift not in use -- a ring of levitation not known cursed, levitation boots known
    not cursed (90% are generated cursed: stuck in the air on a maze level that wasn't the castle), water walking boots,
    an amulet of magical breathing (CFP_MB) with no amulet on. None if there is none."""
    from .power_route import known_cursed
    items = castle._items()
    amulet_on = any(i.category == nh.AMULET_CLASS and i.equipped for i in items)
    boots_on = any(i.is_armor() and i.equipped and 'boots' in (i.text or '') for i in items)
    best = None
    for it in items:
        if it.equipped or not it.is_unambiguous():
            continue
        name = it.object.name
        if it.category == nh.RING_CLASS and name == 'levitation' and not known_cursed(it):
            cand = (0, 'ring', it)
        elif it.is_armor() and name == 'water walking boots' and not known_cursed(it):
            cand = (1, 'boots', it)
        elif it.is_armor() and name == 'levitation boots' and (' uncursed ' in f' {it.text} ' or
                                                               ' blessed ' in f' {it.text} '):
            cand = (2, 'boots', it)
        elif jf_config.CFP_MB and it.category == nh.AMULET_CLASS and name == MB_NAME and not amulet_on:
            cand = (3, 'amulet', it)
        else:
            continue
        if cand[1] == 'boots' and boots_on:
            continue   # (castle_logic._wear_boots takes worn boots off first: a turn more, and not if cursed)
        if best is None or cand[0] < best[0]:
            best = cand
    return None if best is None else (best[1], best[2])


def _early_ring(castle):
    """LIFT_EARLY_RINGS: an unknown ring, not worn and not tried yet, that may be levitation (CFP_RUSH would try it
    right after the recognition dig anyway: here it comes first, before the pit)."""
    for it in castle._items():
        if it.category == nh.RING_CLASS and not it.equipped and not it.is_unambiguous() and \
                'levitation' in _names(it) and it.glyphs and it.glyphs[0] not in castle._tested:
            return it
    return None


def known_rush_strategy(dive):
    """LIFT_KNOWN_RUSH (see the section comment)."""
    def f():
        agent = dive.agent
        castle = dive.castle
        if not jf_config.LIFT_KNOWN_RUSH:
            yield False
            return
        level = agent.current_level()
        key = level.key()
        st = dive.__dict__.setdefault('_lift_known', {})
        entry = st.get(key)
        now = agent.blstats.time
        if entry is None:
            if not _castle_likely(dive):
                if level.dungeon_number == 0 and agent.blstats.depth >= 25:
                    st[key] = {'state': 'done'}
                yield False
                return
            lift = _known_lasting_lift(castle)
            ring = _early_ring(castle) if lift is None and jf_config.LIFT_EARLY_RINGS else None
            if lift is None and ring is None and jf_config.CL_SENSE and not castle.levitating() and \
                    (_cl_lift_potions(castle) or lift_candidates(castle)):
                # castle-lift CL_SENSE: a kit with lifts to test listens for the castle's doors instead of digging the
                # recognition pit (+6 turns median to 'too hard to dig', then a pit to climb out of): the lift tests
                # start as soon as a door sound confirms the castle, and crossings that reach the moat within 30
                # turns of the landing pass far more often (lift suite: 59/157 vs 12/82 after 60 turns)
                yield True
                marked = key not in dive.undiggable
                dive.undiggable.add(key)
                h0 = max(0, len(agent._message_history) - 2)
                st[key] = {'state': 'listen', 't0': now, 'ta': now, 'h0': h0, 'marked': marked}
                _log(castle, f'cl sense: listening for the castle at depth {agent.blstats.depth} '
                             f'(bot x {agent.blstats.x}, Medusa {dive.medusa_level})')
                agent.search()
                return
            if (lift is None and ring is None) or castle.levitating():
                st[key] = {'state': 'done'}
                yield False
                return
            yield True
            marked = key not in dive.undiggable
            # while we test rings or listen afloat the dive must not dig here (a levitating dig can't reach the floor,
            # enough failed tries mark a diggable level undiggable, and a pit costs turns): the level counts as
            # undiggable until we decide
            dive.undiggable.add(key)
            # door sounds are counted from the arrival (the castle's first one within 3 turns of the landing in 101 of
            # 101 arrivals): an early ring test's turns count too (harness lift-er1 heard its doors at +2..+4 while
            # testing rings, then floated at +6 and gave the castle up after 4 more silent turns)
            h0 = max(0, len(agent._message_history) - 2)
            if lift is None:
                st[key] = {'state': 'rings', 'n': 1, 'marked': marked, 'ta': now, 'h0': h0}
                _log(castle, f'early rings: testing {ring.text!r} at depth {agent.blstats.depth} '
                             f'(bot x {agent.blstats.x}, Medusa {dive.medusa_level})')
                castle._try('ring', ring)
                return
            kind, item = lift
            st[key] = {'state': 'on', 't0': now, 'ta': now, 'h0': h0, 'kind': kind, 'marked': marked}
            _log(castle, f'known lift: {kind} {item.text!r} on at once at depth {agent.blstats.depth} '
                         f'(bot x {agent.blstats.x}, Medusa {dive.medusa_level})')
            castle._try(kind, item)
            return
        if entry.get('state') == 'listen':
            heard = [m for m in agent._message_history[entry['h0']:] + [agent.message or '']
                     if any(s in m for s in DOOR_SOUNDS)]
            if heard or castle.castle_key == key:
                entry['state'] = 'done'
                if castle.castle_key != key:
                    inv = '; '.join(f'{agent.inventory.items.get_letter(i)} - {i.text}'
                                    for i in agent.inventory.items.all_items)
                    _log(castle, f'cl sense: {heard[0][:60]!r} +{now - entry["t0"]}: the castle')
                    # (the line the dig failure writes: the kit tools read the castle kit from it)
                    agent.log(f'DIVE bottom reached at depth {agent.blstats.depth}; inventory: {inv}')
                    castle.on_bottom(key)
                dive.undiggable.add(key)
                yield False
                return
            if now >= entry['t0'] + LIFT_LISTEN:
                entry['state'] = 'done'
                if entry.get('marked'):
                    dive.undiggable.discard(key)
                _log(castle, f'cl sense: no door sound in {now - entry["t0"]} turns at depth {agent.blstats.depth}: '
                             f'digging as usual')
                yield False
                return
            if [m for m in agent.get_visible_monsters()
                    if max(abs(m[1] - agent.blstats.y), abs(m[2] - agent.blstats.x)) <= 1]:
                yield False   # something next to us: the usual layers deal with it (we keep listening)
                return
            yield True
            agent.search()
            return
        if entry.get('state') == 'rings':
            if castle.levitating():
                # an unknown ring floated us: levitation, known now -- listen for the castle's doors from here
                entry.update(state='on', t0=now, kind='ring')
                _log(castle, 'early rings: floating')
            else:
                ring = _early_ring(castle) if entry.get('n', 0) < 3 else None
                if ring is None or [m for m in agent.get_visible_monsters()
                                    if max(abs(m[1] - agent.blstats.y), abs(m[2] - agent.blstats.x)) <= 1]:
                    entry['state'] = 'done'
                    if entry.get('marked'):
                        dive.undiggable.discard(key)
                    yield False
                    return
                yield True
                entry['n'] = entry.get('n', 0) + 1
                _log(castle, f'early rings: testing {ring.text!r}')
                castle._try('ring', ring)
                return
        if entry.get('state') != 'on':
            yield False
            return
        if castle.castle_key == key:
            entry['state'] = 'done'   # recognised: CFP_RUSH floats (or walks the moat bottom) from here
            yield False
            return
        if not castle.levitating():
            entry['state'] = 'done'   # water walking / magical breathing: the dive's dig recognises the castle as ever
            if entry.get('marked'):
                dive.undiggable.discard(key)
            yield False
            return
        heard = [m for m in agent._message_history[entry['h0']:] + [agent.message or '']
                 if any(s in m for s in DOOR_SOUNDS)]
        if heard:
            entry['state'] = 'done'
            inv = '; '.join(f'{agent.inventory.items.get_letter(i)} - {i.text}' for i in agent.inventory.items.all_items)
            _log(castle, f'known lift: {heard[0][:60]!r} +{now - entry["t0"]}: the castle')
            # (the line the dig failure writes: the kit tools read the castle kit from it)
            agent.log(f'DIVE bottom reached at depth {agent.blstats.depth}; inventory: {inv}')
            castle.on_bottom(key)
            dive.undiggable.add(key)
            yield False
            return
        if now >= max(entry.get('ta', entry['t0']) + LIFT_LISTEN, entry['t0'] + 2):
            yield True
            entry['state'] = 'done'
            if entry.get('marked'):
                dive.undiggable.discard(key)
            _log(castle, f'known lift: no door sound in {now - entry.get("ta", entry["t0"])} turns at depth {agent.blstats.depth}: '
                         f'not the castle, coming down to dig')
            castle._stop_levitating()
            return
        if [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - agent.blstats.y), abs(m[2] - agent.blstats.x)) <= 1]:
            yield False   # something next to us: the usual layers deal with it (we keep listening)
            return
        yield True
        agent.search()

    return Strategy(f)
