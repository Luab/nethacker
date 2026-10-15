"""Bounded Sokoban pushes using public movement and boulder observations.

Attempt state belongs to GlobalLogic, so strategy preemption does not renew a
blocked push's allowance. Puzzle boulders are never dug, zapped, or picked up.
"""
from dataclasses import dataclass

from nle import nethack as nh

from .exceptions import AgentPanic
from .glyph import G, SS


MAX_TURNS = 80
MAX_PUSHES = 8
MAX_WAITS = 8
MAX_ENTRIES = 64
MAX_APPROACHES = 80
FILLABLE = frozenset((SS.S_pit, SS.S_spiked_pit, SS.S_hole, SS.S_trap_door))


@dataclass
class PushAttempt:
    started: int
    entries: int = 0
    pushes: int = 0
    waits: int = 0
    approaches: int = 0
    pending: bool = False
    outcome: str | None = None
    target_was_hole: bool = False
    retry_wait: bool = False
    completed: bool = False
    progress_noted: bool = False
    failed: str | None = None


def _fail(global_logic, attempt, reason):
    if attempt.failed is None:
        attempt.failed = reason
        global_logic.agent.log(f'SOKOBAN push abandoned: {reason}')
        route = getattr(global_logic, 'wizard_route', None)
        if route is not None and route.active:
            route.solver_failed(reason)
    raise AgentPanic(f'sokoban push: {attempt.failed}')


def _position(agent):
    return int(agent.blstats.y), int(agent.blstats.x)


def _note_push(global_logic, attempt):
    if attempt.progress_noted:
        return
    attempt.progress_noted = True
    route = getattr(global_logic, 'wizard_route', None)
    note = getattr(route, 'note_sokoban_push', None)
    if note is not None:
        note()


def _outcome(agent, key, stance, source, target, attempt):
    if agent.current_level().key() != key:
        return 'level changed during push'
    position = _position(agent)
    source_glyph, target_glyph = int(agent.glyphs[source]), int(agent.glyphs[target])
    if position == source and source_glyph not in G.BOULDER:
        if target_glyph in G.BOULDER or (attempt.target_was_hole and target_glyph in G.FLOOR):
            return 'moved'
        return 'destination boulder or filled hole not observed'
    if position != stance:
        return 'unexpected player displacement'
    if source_glyph not in G.BOULDER:
        return 'boulder changed without player advance'
    return 'blocked'


def _wait(global_logic, attempt):
    if attempt.waits >= MAX_WAITS:
        _fail(global_logic, attempt, 'wait allowance exhausted')
    attempt.waits += 1  # Reserve before SEARCH, whose update can preempt us.
    global_logic.agent.search()


def push_boulder(global_logic, ty, tx, dy, dx):
    """Approach and push once; return True only after a verified outcome.

    Definitive failure raises AgentPanic and informs the optional preparation route.
    The route's solver_failed method handles active preparation-route failure.
"""
    agent = global_logic.agent
    stance = int(ty), int(tx)
    direction = int(dy), int(dx)
    source = stance[0] + direction[0], stance[1] + direction[1]
    target = source[0] + direction[0], source[1] + direction[1]
    key = agent.current_level().key()
    attempts = getattr(global_logic, '_sokoban_push_attempts', None)
    if attempts is None:
        attempts = global_logic._sokoban_push_attempts = {}
    push_key = key, stance, direction
    attempt = attempts.setdefault(push_key, PushAttempt(int(agent.blstats.time)))
    height, width = agent.glyphs.shape
    if abs(direction[0]) + abs(direction[1]) != 1 or not all(
            0 <= y < height and 0 <= x < width for y, x in (stance, source, target)):
        _fail(global_logic, attempt, 'invalid push geometry')
    if attempt.failed is not None:
        _fail(global_logic, attempt, attempt.failed)
    if attempt.completed:
        if int(agent.glyphs[source]) not in G.BOULDER:
            return True  # A completed atomic push was preempted on return.
        # Another boulder has reached a square used by an earlier successful
        # push. This is a new puzzle move, not a renewed failure allowance.
        attempt = attempts[push_key] = PushAttempt(int(agent.blstats.time))
    attempt.entries += 1

    while True:
        if agent.current_level().key() != key:
            _fail(global_logic, attempt, 'level changed while pushing or approaching')
        # Capture was made inside the atomic action, before exit callbacks can
        # move the hero. Reconcile a step interrupted before capture as well.
        if attempt.pending:
            outcome = attempt.outcome or _outcome(agent, key, stance, source, target, attempt)
            attempt.pending, attempt.outcome = False, None
            if outcome == 'moved':
                attempt.completed = True
                _note_push(global_logic, attempt)
                agent.log(f'SOKOBAN pushed {source} toward {target}; attempts={attempt.pushes}')
                return True
            if outcome != 'blocked':
                _fail(global_logic, attempt, outcome)
            attempt.retry_wait = True

        if int(agent.blstats.time) - attempt.started >= MAX_TURNS or \
                attempt.entries > MAX_ENTRIES or attempt.approaches >= MAX_APPROACHES:
            _fail(global_logic, attempt, 'push deadline exhausted')
        prop = agent.character.prop
        if any(getattr(prop, name, False) for name in ('blind', 'hallu', 'confusion', 'stun', 'polymorph')) or \
                int(agent.blstats.prop_mask) & nh.BL_MASK_LEV:
            _fail(global_logic, attempt, 'cannot safely push while impaired or levitating')
        if int(agent.glyphs[source]) not in G.BOULDER:
            _fail(global_logic, attempt, 'expected boulder is not visible')
        if attempt.pushes >= MAX_PUSHES:
            _fail(global_logic, attempt, 'push allowance exhausted')
        if attempt.retry_wait:
            attempt.retry_wait = False
            _wait(global_logic, attempt)
            continue
        if _position(agent) != stance:
            if agent.bfs()[stance] < 0:
                _wait(global_logic, attempt)
            else:
                attempt.approaches += 1
                agent.go_to(*stance, max_steps=1)
            continue

        target_glyph = int(agent.glyphs[target])
        remembered_terrain = int(agent.current_level().objects[target])
        if target_glyph in G.MONS or target_glyph in G.PETS or \
                (target_glyph in G.INVISIBLE_MON and
                 (target_glyph != nh.GLYPH_INVISIBLE or remembered_terrain not in G.FLOOR)):
            _wait(global_logic, attempt)
            continue
        if target_glyph == nh.GLYPH_INVISIBLE and not attempt.waits:
            # An I glyph remembers an unseen creature; it need not still be
            # there. On ordinary floor moverock refuses a real occupant and a
            # successful push clears a stale marker. Never apply this to a pit,
            # where the engine can crush an occupant. Reserve one real wait
            # before rechecking; subsequent refused pushes already wait below.
            _wait(global_logic, attempt)
            continue
        if target_glyph in G.WALL or target_glyph in G.BARS or \
                target_glyph in G.DOOR_CLOSED or target_glyph in G.BOULDER:
            _fail(global_logic, attempt, 'push destination is blocked by terrain or boulder')
        terrain = target_glyph if target_glyph in G.FLOOR or target_glyph in G.TRAPS \
            else remembered_terrain
        if terrain not in G.FLOOR and terrain not in FILLABLE:
            _fail(global_logic, attempt, 'push destination terrain is not confirmed safe')
        attempt.target_was_hole = terrain in FILLABLE
        attempt.pushes += 1
        attempt.pending, attempt.outcome = True, None
        with agent.atom_operation():
            agent.direction(agent.calc_direction(*stance, *source))
            attempt.outcome = _outcome(agent, key, stance, source, target, attempt)
            if attempt.outcome == 'moved':
                _note_push(global_logic, attempt)
