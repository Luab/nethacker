"""CASTLE_TREASURY (jf_config; imported from vlomshakov f84a81b's castle_treasury.py, elite scan F075): while we are a
wall-walker on the castle level (CFP_XORN / POLY_XORN), a bounded detour through the walls to the four tower cells
before the trap door: the castle's wand of wishing is in a chest in one of them.

NetHack 3.6.6 dat/castle.des: CONTAINER:('(',"chest"),not_trapped,$place[0] { OBJECT:('/',"wishing") } with
$place = { (04,02),(58,02),(04,14),(58,14) } shuffled, a burned Elbereth and a cursed scroll of scare monster on the
same square, two soldiers per tower (03/05 and 57/59 of rows 02 and 14). Nothing else puts a chest in a tower, so a
wand in a tower chest IS the wand of wishing (mkobj.c: rnd(3) charges). mksobj() locks a chest 4 times in 5.

Opening a locked chest, best first:
  * a skeleton key / lock pick / credit card (lock.c pick_lock: 'In what direction?' '.', then 'There is a chest
    here; unlock it?' -> y; an occupation);
  * #force with the wielded weapon (lock.c doforce): a blade pries at 2x its large-monster damage per turn (a long
    sword 24%) and breaks 1 time in ~125 per failed turn -- never an artifact (obj_resists 99); a blunt weapon or a
    pick-axe bashes at 2x its die (d3: 6%) and wrecks the box 1 time in 3 (breakchestlock: its wand drops to the
    floor). A sword in the pack is wielded first;
  * kick it from a neighbouring square (dokick.c: the lock breaks 1 kick in 5).
The wand is identified by one zap: its wish is WISH_TELEPORT_ROUTE's (SPARE_WISHES -> power.wish_text ->
tele_route.route_wish: with WISH_CHARGING_FIRST '2 blessed scrolls of charging', else the ring of teleport control),
and the route does the rest (tele_route: a wall-walker may use it with CASTLE_TREASURY on). vlomshakov's ascension
wish list is NOT imported: it switched that route off.

Budgets: 240 actions / 1500 steps / 400 turns, HP >= 45% (and >= 15); trap-door squares are never on the way; a
monster on the way ends the detour (the xorn walk goes around it). Any failure hands back to the trap-door walk.
"""

import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import objects as O
from .exceptions import AgentChangeStrategy, AgentFinished, AgentPanic

TOWERS = ((4, 2), (4, 14), (58, 2), (58, 14))   # castle map (x, y)
# Only the east towers, and only the first one checked: harness oct3 (all four, 27 xorn games) found no wand and cut
# Valley arrivals 18 -> 13 of 30 -- each west tower costs a moat crossing next to a corner shark (castle.des sharks at
# (05,00)/(05,16)/(57,00)/(57,16)) and its two soldiers, from a ~38 HP xorn form. An east tower is ~20 steps off the
# trap-door walk and can be entered from the castle's own corridor (its closed door is no obstacle to a wall-walker)
SEARCH = ((58, 2), (58, 14))
MAX_TOWERS = 1
MOAT_COST = 60                  # vs 20 on the trap-door walk: the corridor way in, not the corner moat
MAX_ACTIONS = 240
MAX_STEPS = 1500
MAX_TURNS = 400
MAX_STALL = 10
MAX_OPEN_TRIES = 30
MIN_HP = 15
MIN_HP_FRACTION = 0.45
UNLOCKERS = ('skeleton key', 'lock pick', 'credit card')
_DIR = {(-1, 0): 'n', (1, 0): 's', (0, 1): 'e', (0, -1): 'w', (-1, 1): 'ne', (-1, -1): 'nw', (1, 1): 'se',
        (1, -1): 'sw'}


class TreasuryState:
    def __init__(self, turn, step):
        self.started_turn = turn
        self.started_step = step
        self.visited = set()        # tower cells checked (no chest, or its chest had no wand)
        self.open_tries = {}        # tower cell -> opening attempts
        self.target = None
        self.actions = 0
        self.same_position = 0
        self.kick_target = None     # tower cell whose locked chest we stepped off to kick
        self.wield_failed = False   # our form couldn't wield the sword: #force with what we hold
        self.acquired_glyph = None
        self.acquired = False
        self.identified = False
        self.done = False
        self.reason = ''


def _log(castle, message):
    castle.agent.log(f'CASTLE TREASURY {message}')


def _finish(castle, state, reason):
    state.done = True
    state.reason = reason
    _log(castle, f'{reason}; back to the trap-door walk')
    return False


def _is_wishing(item):
    return item.category == nh.WAND_CLASS and item.is_unambiguous() and item.object.name == 'wishing'


def _wands(items, glyph=None):
    return [i for i in items if i.category == nh.WAND_CLASS and (glyph is None or glyph in i.glyphs)]


def _identify_acquired(castle, state):
    """One zap names the wand: a wand of wishing asks for the wish at once (NODIR; the route's text answers it)."""
    agent = castle.agent
    inv = agent.inventory
    wand = next(iter(_wands(inv.items, state.acquired_glyph)), None)
    if wand is None:
        return _finish(castle, state, 'the tower wand is no longer carried')
    if not _is_wishing(wand):
        letter = inv.items.get_letter(wand)
        agent._last_wand_use_step = agent.step_count   # tele_route._wand_source: this wish comes from a wand
        _log(castle, f'zapping {wand.text!r} ({letter}) to name it')
        with agent.atom_operation():
            agent.step(A.Command.ZAP)
            if 'What do you want to zap?' in agent.single_message:
                agent.type_text(letter)
            if 'In what direction?' in agent.single_message:
                agent.step(A.Command.ESC)
        _log(castle, f'zap -> {(agent.message or "")[:160]!r}')
        inv.items.update(force=True)
        wand = next(iter(_wands(inv.items, state.acquired_glyph)), None)
    state.identified = wand is not None and _is_wishing(wand)
    _finish(castle, state, f'wand of wishing in the pack: {wand.text!r}' if state.identified else
            'wand taken; the zap did not name it wishing')
    return True


def _blade(item):
    """A weapon #force pries with (lock.c is_blade: dagger..saber skills), and its chance per turn (2x large die)."""
    obj = item.objs[0] if item.objs else None
    if not isinstance(obj, O.Weapon) or not (O.P_DAGGER <= obj.sub <= O.P_SABER) or obj.sub == O.P_PICK_AXE:
        return None
    try:
        return 2 * int(obj.ldam)
    except (TypeError, ValueError):
        return None


def _artifact(item):
    """An artifact's display name is capitalised ('the blessed rustproof +1 Excalibur'); a 'named'/'called' part isn't."""
    text = re.split(r' (?:named|called) ', item.text or '')[0]
    return any(c.isupper() for c in text)


def _can_force_with(item):
    """lock.c doforce: a wielded weapon (or weapon-tool) with skill dagger..lance but not flail; blunt ones bash."""
    obj = item.objs[0] if item.objs else None
    sub = getattr(obj, 'sub', None)
    if not isinstance(obj, (O.Weapon, O.WepTool)) or not isinstance(sub, int):
        return False
    return O.P_DAGGER <= abs(sub) <= O.P_LANCE and abs(sub) != O.P_FLAIL


def _open(castle, state, chest):
    """One step toward opening the locked chest under us: unlock, else #force (a sword wielded first), else kick."""
    agent = castle.agent
    inv = agent.inventory
    pos = castle._pos()
    n = state.open_tries.get(pos, 0)
    if n >= MAX_OPEN_TRIES:
        state.visited.add(pos)
        return _finish(castle, state, f'the tower chest at {pos} stayed locked ({n} tries)')
    state.open_tries[pos] = n + 1
    tools = [i for i in inv.items if i.is_unambiguous() and i.object.name in UNLOCKERS]
    if tools:
        tool = min(tools, key=lambda i: UNLOCKERS.index(i.object.name))
        letter = inv.items.get_letter(tool)

        def responses():
            if 'What do you want to use or apply?' not in agent.single_message:
                return
            yield letter
            for _ in range(3):
                msg = agent.single_message or ''
                if 'In what direction?' in msg or 'direction' in msg.lower():
                    yield '.'
                    continue
                if ('unlock it?' in msg or 'pick its lock?' in msg) and '[yn' in msg:
                    yield 'y'
                    return
                return

        with agent.atom_operation():
            agent.step(A.Command.APPLY, responses())
        _log(castle, f'unlock try {n + 1} at {pos} with {tool.text!r}: {(agent.message or "")[-120:]!r}')
        return True
    wielded = inv.items.main_hand
    # an artifact last among equals: artifact.c touch_artifact blasts a wielder whose alignment record went negative
    # (oct5on s4: 'You are blasted by Excalibur's power!', 4d10 off a 38-HP xorn form)
    swords = sorted((i for i in inv.items if _blade(i) is not None and i.status != i.CURSED),
                    key=lambda i: (-_blade(i), _artifact(i)))
    if swords and not state.wield_failed and (wielded is None or _blade(wielded) is None) and \
            not (wielded is not None and wielded.status == wielded.CURSED):
        if not _wield(castle, state, swords[0]):
            state.wield_failed = True   # (then #force with what we hold)
        return True
    if wielded is not None and _can_force_with(wielded):
        with agent.atom_operation():
            agent.step(A.Command.FORCE)
            for _ in range(2):
                msg = agent.single_message or ''
                if 'force its lock?' in msg or 'force the lock' in msg:
                    agent.type_text('y')
                    break
                if '--More--' in msg:
                    agent.step(A.TextCharacters.SPACE)
                    continue
                break
        _log(castle, f'#force try {n + 1} at {pos} with {wielded.text!r}: {(agent.message or "")[-140:]!r}')
        inv.items.update(force=True)
        return True
    return _kick(castle, state, pos)


def _wield(castle, state, item):
    """Wield a sword to pry with. inventory.wield() refuses in any polymorphed form (its TODO); a xorn has hands
    (harness oct4on s4/s18: 30 refused wields, the locked chest left alone)."""
    agent = castle.agent
    inv = agent.inventory
    letter = inv.items.get_letter(item)
    _log(castle, f'wielding {item.text!r} ({letter}) to pry the lock')
    with agent.atom_operation():
        agent.step(A.Command.WIELD)
        if 'What do you want to wield' not in (agent.single_message or ''):
            if agent._observation['misc'][0] or agent._observation['misc'][1]:
                agent.step(A.Command.ESC)
            return False
        agent.type_text(letter)
    inv.items.update(force=True)
    held = inv.items.main_hand
    ok = held is not None and inv.items.get_letter(held) == letter
    _log(castle, f'wield -> {(agent.message or "")[-100:]!r} ({"ok" if ok else "failed"})')
    return ok


def _kick(castle, state, pos):
    """No key, no weapon to force it with: step off the chest (a wall-walker can stand in the wall beside it); the next
    step kicks back at it (dokick.c: the lock breaks 1 kick in 5) and then walks onto it to look again."""
    agent = castle.agent
    from .castle_logic import to_bot
    cy, cx = (int(v) for v in to_bot(*pos))
    for (dy, dx), d in _DIR.items():
        if castle._monster_at(*_to_map(cy + dy, cx + dx)):
            continue
        agent.direction(d)
        if (int(agent.blstats.y), int(agent.blstats.x)) != (cy, cx):
            state.kick_target = pos
            return True
    return _finish(castle, state, 'no square to kick the tower chest from')


def _to_map(y, x):
    from .castle_logic import to_map
    return to_map(y, x)


def _inspect(castle, state):
    agent = castle.agent
    inv = agent.inventory
    pos = castle._pos()
    inv.get_items_below_me()
    below = list(inv.items_below_me)
    loose = _wands(below)
    if loose and inv.items.free_slots() >= 1:
        # a bashed box scattered its contents (breakchestlock): the wand lies here
        glyph = loose[0].glyphs[0]
        before = sum(i.count for i in _wands(inv.items, glyph))
        inv.pickup(loose[0])
        inv.items.update(force=True)
        if sum(i.count for i in _wands(inv.items, glyph)) > before:
            state.acquired, state.acquired_glyph = True, glyph
            _log(castle, f'picked up the wand at tower {pos}')
            return True
    chests = [i for i in below if i.is_unambiguous() and i.object.name in ('chest', 'large box')]
    if len(chests) != 1:
        state.visited.add(pos)
        state.target = None
        _log(castle, f'tower {pos}: {len(chests)} chests here')
        return True
    chest = chests[0]
    inv.check_container_content(chest)
    if chest.content is None:
        return _finish(castle, state, 'could not look into the tower chest')
    if chest.content.locked:
        return _open(castle, state, chest)
    wands = [i for i in chest.content.items if i.category == nh.WAND_CLASS]
    if not wands:
        state.visited.add(pos)
        state.target = None
        _log(castle, f'tower {pos}: its chest holds no wand')
        return True
    if inv.items.free_slots() < 1:
        return _finish(castle, state, 'no inventory slot for the tower wand')
    wand = next((i for i in wands if _is_wishing(i)), wands[0])
    glyph = wand.glyphs[0]
    before = sum(i.count for i in _wands(inv.items, glyph))
    inv.use_container(chest, [], [wand])
    inv.items.update(force=True)
    if sum(i.count for i in _wands(inv.items, glyph)) <= before:
        return _finish(castle, state, '#loot did not put the tower wand in the pack')
    state.acquired, state.acquired_glyph = True, glyph
    _log(castle, f'took the wand from the chest at tower {pos}')
    return True


def step(castle):
    """True iff the treasury acted this step; False hands back to the trap-door walk at once."""
    from . import castle_cross

    agent = castle.agent
    state = getattr(castle, '_treasury', None)
    if state is not None and state.done:
        return False
    if agent.current_level().key() != castle.castle_key or not castle_cross.wallwalker(agent):
        if state is not None:
            _finish(castle, state, 'left the castle or the wall-walking form')
        return False
    if state is None:
        if any(_is_wishing(i) for i in agent.inventory.items):
            return False   # a wand of wishing already: no detour
        state = castle._treasury = TreasuryState(agent.blstats.time, agent.step_count)
        _log(castle, 'to the towers for the wand of wishing')
    bl = agent.blstats
    if bl.hitpoints < max(MIN_HP, MIN_HP_FRACTION * bl.max_hitpoints):
        return _finish(castle, state, f'HP {bl.hitpoints}/{bl.max_hitpoints}')
    if state.actions >= MAX_ACTIONS or agent.step_count - state.started_step >= MAX_STEPS or \
            bl.time - state.started_turn >= MAX_TURNS:
        return _finish(castle, state, 'budget spent')
    state.actions += 1
    try:
        if state.acquired:
            return _identify_acquired(castle, state)
        pos = castle._pos()
        kt, state.kick_target = state.kick_target, None
        if kt is not None:
            ky, kx = (int(v) for v in castle_cross.to_bot(*kt))
            if max(abs(ky - int(bl.y)), abs(kx - int(bl.x))) == 1:
                agent.kick(ky, kx)
                _log(castle, f'kicked the chest at {kt}: {(agent.message or "")[-120:]!r}')
                return True
        if pos in SEARCH and pos not in state.visited:
            return _inspect(castle, state)
        remaining = set(SEARCH) - state.visited
        if not remaining or len(state.visited & set(SEARCH)) >= MAX_TOWERS:
            return _finish(castle, state, f'tower(s) {sorted(state.visited)} checked, no wand')
        goals = {state.target} if state.target in remaining else remaining
        path = castle_cross._xorn_path(castle, pos, goals=goals, blocked=castle_cross.TRAPDOORS, moat_cost=MOAT_COST,
                                       moat_side_cost=20)
        if not path or len(path) < 2:
            return _finish(castle, state, 'no way to a tower')
        state.target = path[-1]
        nxt = path[1]
        y, x = castle_cross.to_bot(*nxt)
        if castle._monster_at(*nxt):
            return _finish(castle, state, f'a monster on the way at {nxt}')
        if state.actions == 1 or state.actions % 10 == 0:
            _log(castle, f'at {pos}, {len(path) - 1} steps to tower {path[-1]}')
        agent.direction(agent.calc_direction(bl.y, bl.x, y, x))
        after = castle._pos()
        state.same_position = state.same_position + 1 if after == pos else 0
        if state.same_position >= MAX_STALL:
            _finish(castle, state, f'stalled at {after}')
        return True
    except (AgentChangeStrategy, AgentFinished):
        raise
    except AgentPanic:
        _finish(castle, state, 'panic in the container or on the way')
        raise
    except Exception as exc:
        # an optional detour must not end a live game: the recovery loop cancels any menu, the walk resumes
        _finish(castle, state, f'failed: {type(exc).__name__}: {exc}')
        raise AgentPanic('castle treasury failed') from exc
