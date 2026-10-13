"""power's additions to the castle passage (castle_logic.CastlePassage calls these hooks via _power_hook).

Measured on castlebench over all 61 real XL-8-regime arrival kits (pwc-base, CASTLE_PASSAGE on): 0/61 reached
the Valley, 47 died before trying a single item (27 to minotaurs, most within 30 turns of the landing: the
castle's west maze gets rn2(2) minotaurs per MAZEWALK, sp_lev.c fill_empty_maze), 4 never reached the courtyard
test square. True kit contents (--reveal): a levitation/cold/breathing/wishing item in 16, a polymorph source in
12 (potion 4, wand 5, ring 5), a scroll of scare monster in 12.

  CASTLE_ARRIVAL_DRILL  on the landing square, before the walk to the courtyard: engrave-test the unknown wands
                        (nothing hostile close), drop every scroll that may be scare monster (a minotaur won't
                        melee us on it: monmove.c onscary has no minotaur exception for the scroll), then try
                        the passage plan right there (rings, boots, amulets, wishes, potions).
  CASTLE_POLY           a polymorph form that flies, swims or breathes water crosses the moat like levitation
                        (27.5% of the 273 polyok forms; per random polymorph P = 0.2 after system shock and
                        newman, polyself.c); a known wand of polymorph is zapped at ourselves until one comes
                        up; at the door and over the trap door a form that can't kick / fall waits for the
                        form to end (rn1(500, 500) turns, shortened when the form outranks us).
"""

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config, power
from . import objects as O

M1_FLY, M1_SWIM, M1_AMPHIBIOUS = 0x1, 0x2, 0x200
M1_NOHANDS, M1_NOLIMBS, M1_SLITHY = 0x2000, 0x6000, 0x80000
MZ_SMALL = 1
S_LIZARD = None   # the lizard class can't kick (dokick.c); looked up lazily

POLY_WAND = O.from_name('polymorph', nh.WAND_CLASS)
POLY_POTION = O.from_name('polymorph', nh.POTION_CLASS)


# ---------------------------------------------------------------------------------------------- forms

def current_form(agent):
    """permonst of our polymorphed form, None in our own form (Property.polymorph: our square shows a
    different monster glyph than our own)."""
    y, x = agent.blstats.y, agent.blstats.x
    g = agent.glyphs[y, x]
    self_glyph = getattr(agent.character, 'self_glyph', None)
    if self_glyph is None or not nh.glyph_is_monster(g) or g == self_glyph:
        return None
    return nh.permonst(nh.glyph_to_mon(g))


def crosses(p):
    return bool(p.mflags1 & (M1_FLY | M1_SWIM | M1_AMPHIBIOUS))


def flies(p):
    return bool(p.mflags1 & M1_FLY)


def can_kick(p):
    """dokick.c: no legs (nolimbs, slithy), very small, or a lizard can't kick."""
    if (p.mflags1 & M1_NOLIMBS) == M1_NOLIMBS or p.mflags1 & M1_SLITHY or p.msize < MZ_SMALL:
        return False
    return p.mname not in ('lizard', 'newt', 'gecko', 'iguana', 'baby crocodile', 'crocodile', 'chameleon')


def crossing_form(passage):
    """Hook: our form crosses water (castle_logic._floating)."""
    if not jf_config.CASTLE_POLY:
        return False
    p = current_form(passage.agent)
    return p is not None and crosses(p)


def _poly_wand(agent):
    for it in agent.inventory.items:
        if it.is_wand() and it.is_unambiguous() and it.object == POLY_WAND and not power._empty(agent, it):
            return it
    return None


def _zap_self(passage, wand):
    agent = passage.agent
    passage._log(f'power: zapping {wand.text!r} at ourselves')
    agent.zap(wand, '.')
    passage._log(f'power: polymorph zap -> {agent.message!r}')
    if 'Nothing happens' in agent.message or 'You wrest' in agent.message:
        agent.inventory.empty_wands.add(wand.text)
    agent.inventory.items.update(force=True)
    p = current_form(agent)
    if p is not None:
        passage._log(f'power: now a {p.mname} (crosses water: {crosses(p)})')


# ---------------------------------------------------------------------------------------------- drill

def _hostiles_within(agent, radius):
    y0, x0 = agent.blstats.y, agent.blstats.x
    return [m for m in agent.get_visible_monsters() if max(abs(m[1] - y0), abs(m[2] - x0)) <= radius]


def _engrave_test(passage, item):
    """One engrave test (inventory.wand_engrave_identify's core, without its 8-call throttle). A wand of
    wishing makes its wish here: ask for the passage."""
    agent = passage.agent
    inv = agent.inventory
    im = inv.item_manager
    old = getattr(agent, 'wish_purpose', None)
    agent.wish_purpose = 'passage'
    try:
        with agent.atom_operation():
            types = inv._engrave_single_wand(item)
    finally:
        agent.wish_purpose = old
    if types is None:
        return False
    im._glyph_to_possible_wand_types[item.glyphs[0]] = types
    im._already_engraved_glyphs.add(item.glyphs[0])
    im.possible_objects_from_glyph(item.glyphs[0])
    inv.items.update(force=True)
    passage._log(f'power: engrave-tested {item.text!r}: {[o.name for o in types]}')
    return True


def _untested_wands(agent):
    im = agent.inventory.item_manager
    return [i for i in agent.inventory.items if i.is_wand() and not i.is_unambiguous() and len(i.glyphs) == 1
            and i.glyphs[0] not in im._already_engraved_glyphs and i.comment != 'EMPT' and
            not agent.inventory.is_known_empty(i)]


def arrival_step(passage):
    """Hook, on the castle's west side before castle_logic walks to the courtyard. True: acted this step."""
    agent = passage.agent
    drill, poly = jf_config.CASTLE_ARRIVAL_DRILL, jf_config.CASTLE_POLY
    if not (drill or poly):
        return False
    if passage._floating():
        return False   # castle_logic takes a floating hero to the corner (CASTLE_WEST_DIG digs through the maze)
    t = passage._tries
    pos = passage._pos()
    from .castle_logic import MOAT_EDGE
    here_ok = pos not in MOAT_EDGE    # beside the moat: eels and sharks reach us
    if drill and here_ok:
        # 1. unknown wands: cold, polymorph, digging, a wish (2 turns each; the square must be bare)
        if not _hostiles_within(agent, 2) and not agent.character.prop.blind and \
                agent.can_engrave() and t.get('pw_engrave', 0) < 6:
            wands = [w for w in _untested_wands(agent) if w.glyphs[0] not in t.setdefault('pw_engraved', set())]
            if wands:
                t['pw_engrave'] = t.get('pw_engrave', 0) + 1
                t['pw_engraved'].add(wands[0].glyphs[0])
                if _engrave_test(passage, wands[0]):
                    return True
        # (scrolls that may be scare monster are dropped only when an Elbereth-ignorer closes in: CASTLE_SCARE,
        # dive_logic.gehennom_scare -- dropped here at the landing they stayed behind when we moved on)
    cfp_wait = False
    if jf_config.CFP_RUSH:
        # castle-first-pass: the kit's own lasting lifts and the magical-breathing water test go first; a big form
        # tears the armour off (cfp-p2-jf25 s3: a leocrotta form, then a lynx killed the naked dwarf)
        from . import castle_cross
        cfp_wait = castle_cross.pending(passage)
    xorn = False
    if jf_config.CFP_XORN:
        from . import castle_cross
        xorn = castle_cross.wallwalker(agent)
    if poly and not cfp_wait and not xorn:
        # 3. a known wand of polymorph: zap ourselves until a form that crosses water comes up
        form = current_form(agent)
        if jf_config.CFP_MB:
            # castle-first-pass: the form from the messages (the glyph test misses it when we're invisible), and a
            # breathless form counts as crossing -- it walks the moat bottom (mondata.h amphibious() includes
            # M1_BREATHLESS): arm cfpc-s4's giant mimic form walked the moat bottom to the strip; the old test (fly /
            # swim / M1_AMPHIBIOUS) zapped such forms away again
            from . import castle_cross
            form = castle_cross.form_permonst(agent)
            crossing = form is not None and (crosses(form) or castle_cross.amphibious(passage))
        else:
            crossing = form is not None and crosses(form)
        wand = _poly_wand(agent)
        if wand is not None and not crossing and t.get('pw_polyzaps', 0) < 12 and \
                jf_config.CFP_XORN and castle_cross.poly_prep(passage):
            # castle-first-pass: the rings that may be polymorph control go on first (POLY_XORN answers 'xorn')
            return True
        if wand is not None and not crossing and t.get('pw_polyzaps', 0) < 12:
            if form is not None and agent.blstats.carrying_capacity >= 4:
                # a small form is Overtaxed by our pack and zap.c check_capacity refuses to zap: keep only the wand
                rest = [i for i in agent.inventory.items if i is not wand and i.can_be_dropped_from_inventory()
                        and i.category != nh.COIN_CLASS]
                if rest and not t.get('pw_polydrop'):
                    t['pw_polydrop'] = 1
                    passage._log(f'power: {form.mname} form overloaded: dropping {len(rest)} items to zap again')
                    agent.inventory.drop(rest, smart=False)
                    return True
            t['pw_polyzaps'] = t.get('pw_polyzaps', 0) + 1
            t.pop('pw_polydrop', None)
            _zap_self(passage, wand)
            return True
        if form is None and t.get('pw_polyzaps') and agent.inventory.items_below_me and \
                not t.get('pw_repick'):
            # back in our own form on the pile a small form dropped: take it back (not the drill's scrolls)
            t['pw_repick'] = 1
            agent.inventory.pickup_and_drop_items().run()
            return True
    if drill and here_ok:
        # 4. the lasting lifts of the passage plan right here instead of after the walk through the maze (a
        # potion's 10-149 turns of levitation are kept for the moat: castle_logic quaffs them by the corner)
        plan = passage._plan()
        if plan and not passage._resting:
            kind, item = plan[0]
            # known lifts are castle_logic's (it rests first); the drill tests unknown ones where we land (a known
            # ring taken off for a rest stop was put straight back on here: pwc-dp6 jf16-s13 looped on it)
            if kind in ('ring', 'boots', 'amulet', 'wish') and (kind == 'wish' or not item.is_unambiguous()):
                passage._set_state(f'power drill: trying {kind} {item.text!r} where we landed')
                passage._try(kind, item)
                return True
    return False


# ---------------------------------------------------------------------------------------------- door

def door_step(passage, pos):
    """Hook at the back door / trap door. True while a polymorph form can't do the door work (no legs to
    kick and no known wand to zap, or flying over the trap door): wait for the form to end out of the eels'
    reach, fighting what comes."""
    if not jf_config.CASTLE_POLY:
        return False
    agent = passage.agent
    p = current_form(agent)
    if p is None:
        return False
    from .castle_logic import DOOR, TRAPDOOR, SAFE_EAST, OUTSIDE, _bfs
    if jf_config.BREACH_PLUNGE and pos == TRAPDOOR and flies(p) and not passage.levitating():
        # do.c dodown: on a trap door we have seen (flying over it: 'A trap door opens up under you! You don't fall
        # in.' -- trap.c feeltrap) '>' plunges through it even while flying (dotrap TOOKPLUNGE); only levitation
        # refuses. The old rule waited up to 3000 turns for the form to end here.
        t = passage._tries
        t['pw_plunge'] = t.get('pw_plunge', 0) + 1
        if t['pw_plunge'] <= 3:
            passage._set_state(f'power: {p.mname} form plunging through the trap door')
            agent.direction('>')
            passage._log(f"'>' on the trap door as a {p.mname}: {agent.message!r}")
            return True
    wait = False
    if pos in (DOOR, TRAPDOOR) and flies(p):
        wait = True                       # a flyer floats over the trap door (trap.c: Flying)
    elif not passage._door_open() and not can_kick(p) and \
            not any(passage._usable_wand(i, n) for i in passage._items() for n in ('striking', 'digging', 'opening')):
        wait = True                       # can't kick, nothing to zap at the lock
    if not wait:
        return False
    t = passage._tries
    t['pw_formwait'] = t.get('pw_formwait', 0) + 1
    if t['pw_formwait'] > 3000:
        return False
    if pos not in (DOOR, TRAPDOOR, SAFE_EAST) and passage._step_downhill(_bfs(SAFE_EAST, OUTSIDE), pos):
        return True
    if not passage._fight_adjacent():
        passage._set_state(f'power: waiting for the {p.mname} form to end')
        agent.search(3)
    return True


# ---------------------------------------------------------------------------------------------- BREACH_MINO

_W = {n: O.from_name(n, nh.WAND_CLASS) for n in (
    'sleep', 'death', 'teleportation', 'polymorph', 'striking', 'fire', 'cold', 'lightning', 'magic missile',
    'slow monster', 'speed monster', 'make invisible', 'create monster', 'digging', 'cancellation')}
# what a zap does to a minotaur (15HD, MR 0, 3d10/3d10/2d8 a turn, speed 15): stops it for good, hurts it, or worse
_DECISIVE = ('sleep', 'death', 'teleportation', 'polymorph')
_DAMAGE = ('striking', 'fire', 'cold', 'lightning', 'magic missile', 'slow monster')
_HARMFUL = ('speed monster', 'make invisible', 'create monster')
MINO_RANGE = 8


def is_big_ignorer(dive, mon):
    """A minotaur, or another Elbereth-ignorer of level 10+ (captain, Elvenking...)."""
    name = getattr(mon, 'mname', '')
    if name == 'minotaur':
        return True
    return dive._ignores_elbereth(mon) and name != 'unknown' and getattr(mon, 'mlevel', 0) >= 10


def breach_wand(agent, unknown_ok=True):
    """BREACH_MINO: the wand to zap at a minotaur, best first: a known wand of sleep/death, teleportation, polymorph,
    then striking/fire/cold/lightning/magic missile/slow monster, then an unknown wand not zapped yet whose possible
    types (price, engrave test) are more likely to stop or hurt it than to help it. (wand, why) or (None, None)."""
    items = [i for i in agent.inventory.items if i.is_wand() and not power._empty(agent, i)]
    for group in (_DECISIVE, _DAMAGE):
        for name in group:
            for it in items:
                if it.is_unambiguous() and it.object == _W[name]:
                    return it, f'known {name}'
    if not unknown_ok:
        return None, None
    zapped = agent._last_resort_zapped
    best = None
    for it in items:
        if it.is_unambiguous() or it.glyphs[0] in zapped or len(it.glyphs) != 1:
            continue
        good = power.p_of(it, {_W[n] for n in _DECISIVE + _DAMAGE})
        bad = power.p_of(it, {_W[n] for n in _HARMFUL})
        decisive = power.p_of(it, {_W[n] for n in _DECISIVE})
        if good <= bad:
            continue
        key = (decisive, good - bad)
        if best is None or key > best[0]:
            best = (key, it)
    if best is None:
        return None, None
    return best[1], f'unknown P(stop)={best[0][0]:.2f} P(good-bad)={best[0][1]:.2f}'


def zap_breach_wand(agent, wand, why, target, direction):
    """Zap `wand` at `target` ((_, y, x, mon, _)) in `direction`; an unknown one is marked tried."""
    _, my, mx, mon, _ = target
    if not wand.is_unambiguous():
        agent._last_resort_zapped.add(wand.glyphs[0])
    agent.log(f'BREACH zapping {wand.text!r} ({why}) at the {getattr(mon, "mname", "?")} at ({my},{mx}) '
              f'hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints}')
    agent.zap(wand, direction)
    msg = agent.message
    agent.log(f'BREACH zap -> {msg[:120]!r}')
    if 'Nothing happens' in msg or 'You wrest' in msg:
        agent.inventory.empty_wands.add(wand.text)
    agent.inventory.items.update(force=True)


def inline_targets(dive, max_dist=MINO_RANGE):
    """Big Elbereth-ignorers in a straight line 2..max_dist squares away over walkable squares:
    [(dist, direction, target)]."""
    agent = dive.agent
    bl = agent.blstats
    level = agent.current_level()
    out = []
    for m in agent.get_visible_monsters():
        _, my, mx, mon, _ = m
        dy, dx = int(my - bl.y), int(mx - bl.x)
        dist = max(abs(dy), abs(dx))
        if dist < 2 or dist > max_dist or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)) or \
                not is_big_ignorer(dive, mon):
            continue
        sy, sx = (dy > 0) - (dy < 0), (dx > 0) - (dx < 0)
        if any(not level.walkable[bl.y + sy * k, bl.x + sx * k] for k in range(1, dist)):
            continue
        out.append((dist, agent.calc_direction(bl.y, bl.x, bl.y + sy, bl.x + sx), m))
    out.sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------------------------- deep escape

def deep_poly_escape_strategy(dive):
    """CASTLE_POLY, from CASTLE_SCARE_DEPTH on the main line (the castle or the mazes right above it, before the
    castle is even recognised): losing a fight to what's next to us with a known wand of polymorph -> zap it at
    ourselves. The new form's hit points are a fresh pool (its death returns us to our own form with the HP we
    had), and one form in five flies, swims or breathes water: the moat crossing (castlebench pwc-dp5: the
    polymorph-wand kits jf16-s14, jf16-s6 died to a minotaur in 7-22 turns while still digging for the castle)."""
    from .level import Level
    from .strategy import Strategy
    from . import utils

    def f():
        agent = dive.agent
        if not jf_config.CASTLE_POLY:
            yield False
            return
        level = agent.current_level()
        bl = agent.blstats
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or bl.depth < jf_config.CASTLE_SCARE_DEPTH:
            yield False
            return
        form = current_form(agent)
        if form is not None and crosses(form):
            yield False
            return
        from . import castle_cross
        if jf_config.CFP_XORN and castle_cross.wallwalker(agent):
            # castle-first-pass: a wall-walking form (xorn) is the way to the trap doors -- no re-zap (power-route's
            # harness pr-polyx2 zapped the wand again as a xorn at 13/31 HP with a minotaur adjacent)
            yield False
            return
        hp, hpmax = bl.hitpoints, bl.max_hitpoints
        adjacent = [m for m in agent.get_visible_monsters() if utils.adjacent((m[1], m[2]), (bl.y, bl.x))]
        if not adjacent:
            yield False
            return
        if jf_config.BREACH_MINO:
            # at any HP: a minotaur takes 30-50 HP a turn and nothing but a wand (or a scroll of scare monster)
            # stops it; the kit's wand of polymorph goes at IT (the crossing self-zap comes later, CASTLE_POLY)
            big = [m for m in adjacent if is_big_ignorer(dive, m[3])]
            if big and not dive.on_scare_scroll():
                bwand, why = breach_wand(agent)
                if bwand is not None:
                    yield True
                    target = big[0]
                    zap_breach_wand(agent, bwand, why, target,
                                    agent.calc_direction(bl.y, bl.x, target[1], target[2]))
                    return
        wand = _poly_wand(agent) if hp < 0.6 * hpmax else None
        if wand is None:
            # an Elbereth-ignorer (minotaur, @) at us from the landing on: gamble each unknown wand on it now rather
            # than at critical HP (LAST_RESORT) -- sleep, striking, fire/cold/lightning, slow, polymorph... name
            # themselves on a hit; teleportation is inert here (noteleport). 24 of 61 real kits died to a minotaur,
            # most within 10 turns of the landing (castlebench pwc-dp5..dp7)
            ignorers = [m for m in adjacent if dive._ignores_elbereth(m[3])]
            if not ignorers or hp >= 0.9 * hpmax:
                yield False
                return
            zapped = agent._last_resort_zapped
            unknown = [i for i in agent.inventory.items if i.is_wand() and not i.is_unambiguous() and
                       i.glyphs[0] not in zapped and not agent.inventory.is_known_empty(i) and i.comment != 'EMPT']
            if not unknown:
                yield False
                return
            yield True
            it = unknown[0]
            zapped.add(it.glyphs[0])
            _, my, mx, mon, _ = ignorers[0]
            agent.log(f'POWER deep escape: zapping unknown {it.text!r} at the {getattr(mon, "mname", "?")} at {hp}/{hpmax}')
            agent.zap(it, agent.calc_direction(bl.y, bl.x, my, mx))
            agent.log(f'POWER deep escape: {agent.message[:100]!r}')
            agent.inventory.items.update(force=True)
            return
        yield True
        agent.log(f'POWER deep escape: zapping {wand.text!r} at ourselves at {hp}/{hpmax} HP, adjacent '
                  f'{[getattr(m[3], "mname", "?") for m in adjacent]}')
        agent.zap(wand, '.')
        if 'Nothing happens' in agent.message or 'You wrest' in agent.message:
            agent.inventory.empty_wands.add(wand.text)
        agent.inventory.items.update(force=True)
        p = current_form(agent)
        agent.log(f'POWER deep escape: {agent.message[:100]!r} -> form {getattr(p, "mname", None)}')

    return Strategy(f)
