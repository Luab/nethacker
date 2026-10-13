"""Castle passage: from the castle's west landing region down a trap door into the Valley of the Dead.

Everything here is from NetHack 3.6.6 (castle.des, sp_lev.c, mkmaze.c, trap.c, zap.c, dokick.c, lock.c):

* Placement. GEOMETRY center,center puts the 63x17 map at xstart = 2 + (78 - 2 - 63) / 2 = 8 -> odd 9 and
  ystart = 2 + (20 - 2 - 17) / 2 = 2 -> odd 3, so castle map square (mx, my) is level (mx + 9, my + 3), i.e.
  bot/NLE (row my + 3, col mx + 8) (NLE drops level column 0). Checked on a replay: the pick-axe fall of
  public seed 10 landed in the west maze at bot (11, 6).
* Landing. A fall from above lands in levregion(1,0,10,20) minus the castle: the west maze or the west
  courtyard (map x 0-4, y 6-10). That region is sealed: the maze walk never leaves y >= 3 (mkmaze okay()),
  rows <= 2 and the level edge are undiggable (bound_digging), the map is NON_DIGGABLE, and the castle is
  the bottom level (Can_dig_down false: only pits). Its only way on is over the moat.
* Front. The drawbridge (05,08) opens into the antechamber (8 soldiers and a lieutenant, all @, so no
  Elbereth), then the barracks hall, then the throne room (27 court monsters), then a secret door (38,08)
  in front of the trap doors (40..55,08). Hopeless for an XL 8-10 dive.
* Back. The east courtyard (map x 57-62) touches the locked door (56,08) by land, and the trap door
  (55,08) lies right behind it. The east side is reached only over water: the moat ring along map row 0
  (13 moat squares at each end of the dry strip x 9-53; a shark starts on each end).
* So: levitation (ring, potion, boots) or water walking floats/walks the ring; a cold ray (wand of cold,
  frost horn) freezes it into ice (zap.c: ice lasts >= 50 turns). A levitating hero can't kick a door
  (dokick: "not enough leverage") or fall through a trap door, but float_down() over a trap door drops
  us through it -- so we float onto (55,08) and come down there (remove the ring / let the potion end).
* Dying here costs nothing: the score is the maximum depth and the castle is as deep as a dig can go.
  So every untested ring and potion that could be levitation is tried (160 castle arrivals in past runs
  carried 1.7 unknown ring and 2.3 unknown potion types on average: ~16% held a way across).
"""

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from .exceptions import AgentChangeStrategy, AgentFinished, AgentPanic
from .glyph import G, SS
from .level import Level
from .strategy import Strategy

CASTLE_MAP = [
    '}}}}}}}}}.............................................}}}}}}}}}',
    '}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}',
    '}|.....|-----------------------------------------------|.....|}',
    '}|.....+...............................................+.....|}',
    '}-------------------------------+-----------------------------}',
    '}}}}}}|........|..........+...........|.......S.S.......|}}}}}}',
    '.....}|........|..........|...........|.......|.|.......|}.....',
    '.....}|........------------...........---------S---------}.....',
    '.....}|...{....+..........+.........\\.S.................+......',
    '.....}|........------------...........---------S---------}.....',
    '.....}|........|..........|...........|.......|.|.......|}.....',
    '}}}}}}|........|..........+...........|.......S.S.......|}}}}}}',
    '}-------------------------------+-----------------------------}',
    '}|.....+...............................................+.....|}',
    '}|.....|-----------------------------------------------|.....|}',
    '}-------}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}-------}',
    '}}}}}}}}}.............................................}}}}}}}}}',
]
MAP_W, MAP_H = 63, 17
ROW0, COL0 = 3, 8          # bot (row, col) of castle map (0, 0)
CORNER = (0, 6)            # west courtyard corner at the foot of the moat ring: cold rays start here
TEST_SPOT = (1, 7)         # next to the corner with no water beside it: rings and potions are tried here
GOAL = (57, 8)             # east courtyard square in front of the back door
DOOR = (56, 8)             # locked door into the trap-door hall
TRAPDOOR = (55, 8)         # the easternmost of the five trap doors (40,44,48,52,55 on row 8)
SAFE_EAST = (59, 8)        # east courtyard square out of the moat eels' reach
# straight segments for cold rays: up the west column, along row 0, down the east column
ROUTE = ([(0, y) for y in range(6, -1, -1)] + [(x, 0) for x in range(1, 63)] +
         [(62, y) for y in range(1, 7)] + [(61, 7), (60, 8), (59, 8), (58, 8), (57, 8)])
ROUTE_INDEX = {p: i for i, p in enumerate(ROUTE)}

DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def to_bot(mx, my):
    return my + ROW0, mx + COL0


def to_map(y, x):
    return x - COL0, y - ROW0


def map_char(mx, my):
    if 0 <= mx < MAP_W and 0 <= my < MAP_H:
        return CASTLE_MAP[my][mx]
    return ' '


def _outside_squares():
    """Squares a levitator can use outside the castle walls: flood fill from the west courtyard over
    floor and moat (the drawbridge square is moat while raised; doors and walls stop it)."""
    assert len(CASTLE_MAP) == MAP_H and all(len(r) == MAP_W for r in CASTLE_MAP)
    seen = {CORNER}
    todo = [CORNER]
    while todo:
        mx, my = todo.pop()
        for dx, dy in DIRS:
            n = (mx + dx, my + dy)
            if n in seen or map_char(*n) not in '.}':
                continue
            if dx and dy and map_char(mx + dx, my) not in '.}' and map_char(mx, my + dy) not in '.}':
                continue
            seen.add(n)
            todo.append(n)
    return frozenset(seen)


OUTSIDE = _outside_squares()
WEST_COURTYARD = frozenset((x, y) for x in range(0, 5) for y in range(6, 11))
# the northern half of the ring (and both courtyards): one fixed way round. With both halves open, a shark
# revealed on one made the BFS turn back to the other and back again, dancing in the channel under the
# shark's bites and a xorn's in the wall (castle-c2 seed 4)
NORTH = frozenset(p for p in OUTSIDE if p[1] <= 8 or p in WEST_COURTYARD or p[0] >= 57)
# CASTLE_SEA_SWITCH: the mirror way round, taken when a sea monster holds the north half's west column
SOUTH = frozenset(p for p in OUTSIDE if p[1] >= 8 or p in WEST_COURTYARD or p[0] >= 57)
ROUTE_S = frozenset([(0, y) for y in range(10, 17)] + [(x, 16) for x in range(1, 63)] +
                    [(62, y) for y in range(15, 9, -1)] + [(61, 9), (60, 8), (59, 8), (58, 8), (57, 8)])


def west_zone(p):
    """'north' / 'south': a moat square of that half's west end -- the one-square-wide column x 0, row 0/16 up to
    the top/bottom strip, and (1,5)/(1,11) beside the courtyard corner. A sea monster there bites everything
    entering that half. (Row 5/11 further east is where the drawbridge eels pass: not counted.)"""
    x, y = p
    if map_char(x, y) != '}':
        return None
    if (x == 0 and y <= 5) or (y == 0 and x <= 8) or p == (1, 5):
        return 'north'
    if (x == 0 and y >= 11) or (y == 16 and x <= 8) or p == (1, 11):
        return 'south'
    return None
# courtyard squares beside the moat: its eels (start (05,07),(05,09)) and sharks (5d6 bites) reach them
# (jf27 seed 8: fight2 stepped onto the corner to throw daggers at an ape; an eel and a shark took 34 HP)
MOAT_EDGE = frozenset(p for p in WEST_COURTYARD
                      if any(map_char(p[0] + dx, p[1] + dy) == '}' for dx, dy in DIRS))


def _bfs(start, passable):
    """Distances from start over passable map squares (no diagonal squeeze between two walls)."""
    dist = {start: 0}
    todo = [start]
    i = 0
    while i < len(todo):
        mx, my = todo[i]
        i += 1
        for dx, dy in DIRS:
            n = (mx + dx, my + dy)
            if n in dist or n not in passable:
                continue
            if dx and dy and map_char(mx + dx, my) not in '.}' and map_char(mx, my + dy) not in '.}':
                continue
            dist[n] = dist[(mx, my)] + 1
            todo.append(n)
    return dist


def _downhill(dist, pos):
    """Neighbours of pos one step closer to the BFS source."""
    d = dist.get(pos)
    if not d:
        return []
    return [(pos[0] + dx, pos[1] + dy) for dx, dy in DIRS if dist.get((pos[0] + dx, pos[1] + dy)) == d - 1]


class CastlePassage:
    def __init__(self, dive):
        self.dive = dive
        self.agent = dive.agent
        self.castle_key = None
        self.given_up = False
        self._tested = set()           # glyphs of rings/potions/boots/wands already tried
        self._lev_source = None        # ('ring' | 'potion' | 'boots', glyph) that made us float
        self._tries = {}               # attempt counters (door methods, cold rays, waits)
        self._stuck = 0                # crossing steps without moving
        self._last_pos = None
        self._approach_tries = 0
        self._log_state = None
        self._arrival_logged = False
        self._waiting = False          # floating on a potion by the closed door: wait it out at SAFE_EAST
        self._cold_glyphs = set()      # unknown horns that froze the moat when blown
        self._held_turn = None         # last turn a sea monster held us ('swings itself around you')
        self._resting = False          # a rest stop on the way across (lift off, Elbereth, search)
        self._last_dnum = None         # dungeon of the previous step (Gehennom -> Dungeons: the castle)
        self._half = 'north'           # CASTLE_SEA_SWITCH: the way round ('north' | 'south')
        self._sea = {}                 # CASTLE_SEA_SWITCH: moat square -> last turn a monster was seen on it
        self._sea_retreat = False      # CASTLE_SEA_SWITCH: backing out of a held channel to the courtyard

    # ------------------------------------------------------------------ state

    def on_bottom(self, key):
        """dig_with_tool: 'The floor here is too hard to dig in' on the Dungeons of Doom at depth >= 25.
        dungeon.def: the Dungeons have 25-29 levels and the castle is the last; Medusa's levels are
        diggable, so on the main line only the castle refuses a hole."""
        if self.castle_key != key:
            self.castle_key = key
            self._log(f'level detected: {key} (depth {self.agent.blstats.depth})')

    def note_level(self):
        """DiveLogic.update, every step: coming up from Gehennom into the Dungeons means the castle (the
        Valley's '<' is its only way up), even if we never dug into its west side."""
        level = self.agent.current_level()
        dnum = level.dungeon_number
        if self._last_dnum == 1 and dnum == Level.DUNGEONS_OF_DOOM:
            self.castle_key = level.key()
            self.given_up = False
            self._tries = {}
            self._resting = False
            self._stuck = 0
            self._log(f'back from Gehennom: {level.key()} (depth {self.agent.blstats.depth})')
        self._last_dnum = dnum

    def _maybe_resume(self):
        """BREACH_RESUME: a passage given up for want of a way across comes back once one turns up (a wand of cold
        named by a zap at a monster, a lift found by a last-resort quaff, a crossing polymorph form). Checked every
        25 turns, at most 6 times: pwc/brx jf14-s0 gave up with an untested wand that a deep-escape zap then named."""
        why = getattr(self, '_given_up_why', '') or ''
        if not why.startswith(('nothing that crosses water', 'stranded')):
            return
        now = self.agent.blstats.time
        if now - getattr(self, '_resume_check', -10 ** 9) < 25 or getattr(self, '_resumes', 0) >= 6:
            return
        self._resume_check = now
        if self.castle_key is None or self.agent.current_level().key() != self.castle_key:
            return
        if self._floating() or self._plan() or self._cold_source() is not None:
            self._resumes = getattr(self, '_resumes', 0) + 1
            self.given_up = False
            self._stuck = 0
            self._log(f'resuming the passage ({self._resumes}): a way across turned up after "{why}"')

    def active(self):
        if jf_config.BREACH_RESUME and self.given_up and jf_config.CASTLE_PASSAGE:
            self._maybe_resume()
        return jf_config.CASTLE_PASSAGE and self.castle_key is not None and not self.given_up and \
            self.agent.current_level().key() == self.castle_key and \
            self.agent.current_level().dungeon_number == Level.DUNGEONS_OF_DOOM

    def _condition(self, mask):
        return bool(int(self.agent.last_observation['blstats'][nh.NLE_BL_CONDITION]) & mask)

    def levitating(self):
        return self._condition(nh.BL_MASK_LEV)

    def _items(self):
        return list(self.agent.inventory.items)

    @staticmethod
    def _names(item):
        return {getattr(o, 'name', '') for o in item.objs}

    def water_walking(self):
        return any(i.equipped and i.is_unambiguous() and i.object.name == 'water walking boots'
                   for i in self._items())

    def _floating(self):
        if self.levitating() or self.water_walking() or self._power_hook('crossing_form', False):
            return True
        if jf_config.CFP_MB:
            # castle-first-pass: magical breathing (a water-tested amulet) or an amphibious/breathless form walks the
            # moat bottom round the ring like a levitator floats it (castle_cross.amphibious)
            from . import castle_cross
            return castle_cross.amphibious(self)
        return False

    def _power_hook(self, name, default, *args):
        """Optional teammate module castle_power.py (power's arrival drill / polymorph forms). Hooks:
        arrival_step(passage) -> bool (it acted this step), crossing_form(passage) -> bool (a form that flies or
        swims counts as floating). A missing module or hook is `default`; a bug in it must not end the passage."""
        try:
            from . import castle_power
        except ImportError:
            return default
        fn = getattr(castle_power, name, None)
        if fn is None:
            return default
        try:
            return fn(self, *args)
        except (AgentChangeStrategy, AgentFinished, AgentPanic):
            raise
        except Exception as e:
            self._log(f'castle_power.{name} failed: {e!r}')
            return default

    def _pos(self):
        bl = self.agent.blstats
        return to_map(bl.y, bl.x)

    def _log(self, msg):
        self.agent.log(f'CASTLE {msg}')

    def _set_state(self, state):
        if state != self._log_state:
            self._log_state = state
            self._log(f'{state} (pos {self._pos()}, turn {self.agent.blstats.time})')

    def _give_up(self, why):
        if not self.given_up:
            self._log(f'giving up: {why}')
            self._given_up_why = why
        self.given_up = True

    # ------------------------------------------------------------------ items

    def _usable_wand(self, item, name):
        return item.category == nh.WAND_CLASS and item.is_unambiguous() and item.object.name == name and \
            not self.agent.inventory.is_known_empty(item) and item.comment != 'EMPT'

    def _cold_source(self):
        if self._tries.get('cold_done'):
            return None
        for item in self._items():
            if self._usable_wand(item, 'cold') or (item.is_unambiguous() and item.object.name == 'frost horn') \
                    or item.glyphs[0] in self._cold_glyphs:
                return item
        return None

    def _candidates(self):
        """(rank, kind, item) that may lift us over the moat: identified first, then untested."""
        out = []
        for item in self._items():
            names = self._names(item)
            known = item.is_unambiguous()
            if item.category == nh.RING_CLASS and not item.equipped and 'levitation' in names and \
                    (known or item.glyphs[0] not in self._tested):
                out.append((0 if known else 2, 'ring', item))
            elif item.category == nh.POTION_CLASS and 'levitation' in names and \
                    (known or item.glyphs[0] not in self._tested):
                out.append((1 if known else 3, 'potion', item))
            elif item.is_armor() and known and not item.equipped and \
                    item.object.name in ('levitation boots', 'water walking boots'):
                out.append((0, 'boots', item))
        # (a wand of wishing is not zapped here: agent.update answers every wish prompt with gray dragon
        # scale mail; power.passage_plan may ask for a 'wish' step, handled by _wish)
        out.sort(key=lambda t: t[0])
        return out

    def _plan(self):
        """[(kind, item)] to try, in order: a teammate's power.passage_plan (price groups, unknown boots,
        wishes) when that module exists, else our own list."""
        plan = None
        try:
            from . import power
            plan = power.passage_plan(self.agent, self._tested)
        except (ImportError, AttributeError):
            plan = None
        except (AgentChangeStrategy, AgentFinished, AgentPanic):
            raise
        except Exception as e:  # a planning bug must not end the passage
            self._log(f'power.passage_plan failed: {e!r}')
            plan = None
        if plan is None:
            return [(kind, item) for _, kind, item in self._candidates()]
        out = []
        for step in plan:
            action, item = step[0], step[1]
            if item.glyphs[0] in self._tested and not (jf_config.CASTLE_EDGE_REST and item.is_unambiguous()):
                continue
            if action == 'puton':
                out.append(('amulet' if item.category == nh.AMULET_CLASS else 'ring', item))
            elif action == 'quaff':
                out.append(('potion', item))
            elif action == 'wear':
                out.append(('boots', item))
            elif action == 'wish':
                out.append(('wish', item))
            elif action == 'apply' and not (item.is_unambiguous() and item.object.name == 'frost horn'):
                out.append(('horn', item))
            # 'zap' (a known wand of cold) and a known frost horn: _cold_source; 'engrave': the bot's own
            # wand_engrave_identify
        return out

    def _worn_rings(self):
        return [i for i in self._items() if i.category == nh.RING_CLASS and i.equipped]

    def _try(self, kind, item):
        self._set_state(f'trying {kind} {item.text!r}')
        if kind == 'ring':
            self._put_on_ring(item)
        elif kind == 'amulet':
            self._put_on_amulet(item)
        elif kind == 'potion':
            self._quaff(item)
        elif kind == 'boots':
            self._wear_boots(item)
        elif kind == 'wish':
            self._wish(item)
        elif kind == 'horn':
            self._blow_horn(item)
        else:
            self._tested.add(item.glyphs[0])
            self.agent.search()

    def _put_on_amulet(self, item):
        """An unknown amulet may be magical breathing (a moat eel's wrap can't drown us then). Strangulation
        is taken straight off again (6 turns to live)."""
        agent = self.agent
        letter = agent.inventory.items.get_letter(item)
        self._tested.add(item.glyphs[0])

        def gen():
            if 'What do you want to put on?' in agent.single_message:
                yield letter

        with agent.atom_operation():
            agent.step(A.Command.PUTON, gen())
        self._log(f'put on {item.text!r}: {agent.message!r}')
        if 'constricts your throat' in agent.message:
            agent.inventory.items.update(force=True)
            worn = next((i for i in self._items() if i.category == nh.AMULET_CLASS and i.equipped), None)
            if worn is not None:
                self._remove_ring(worn)

    def _blow_horn(self, item):
        """An unknown horn may be a frost horn: blown from the moat-ring corner over the water ahead, ice
        tells (then it is our cold source). A fire horn only boils a little water off the moat."""
        agent = self.agent
        if self._pos() != CORNER:
            self._approach(CORNER)
            return
        self._tested.add(item.glyphs[0])
        nxt = ROUTE[ROUTE_INDEX[CORNER] + 1]
        d = agent.calc_direction(agent.blstats.y, agent.blstats.x, *to_bot(*nxt))
        with agent.atom_operation():
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(item))
            if 'In what direction?' in agent.single_message:
                agent.direction(d)
        self._log(f'blew {item.text!r} {d}: {agent.message!r}')
        if self._dry(*nxt) and map_char(*nxt) == '}':
            self._cold_glyphs.add(item.glyphs[0])
            self._log('the horn froze the moat: a frost horn')

    def _put_on_ring(self, item):
        agent = self.agent
        worn = self._worn_rings()
        if len(worn) >= 2:
            # both fingers busy: take one off (a cursed one stays on; then this ring is skipped)
            spare = next((r for r in worn if 'levitation' not in self._names(r) or not r.is_unambiguous()), None)
            if spare is None or not self._remove_ring(spare):
                self._tested.add(item.glyphs[0])
                agent.search()
            return
        letter = agent.inventory.items.get_letter(item)

        def gen():
            if 'What do you want to put on?' not in agent.single_message:
                return
            yield letter
            if 'Which ring-finger' in agent.single_message:
                yield 'r'

        with agent.atom_operation():
            agent.step(A.Command.PUTON, gen())
        self._tested.add(item.glyphs[0])
        self._log(f'put on {item.text!r}: {agent.message!r}')
        if self.levitating():
            self._lev_source = ('ring', item.glyphs[0])
            return
        # not levitation: take it off again (a cursed one stays on)
        agent.inventory.items.update(force=True)
        ring = next((r for r in self._worn_rings() if r.glyphs[0] == item.glyphs[0]), None)
        if ring is not None:
            self._remove_ring(ring)

    def _remove_ring(self, ring):
        agent = self.agent
        letter = agent.inventory.items.get_letter(ring)

        def gen():
            if 'What do you want to remove?' in agent.single_message:
                yield letter

        with agent.atom_operation():
            agent.step(A.Command.REMOVE, gen())
        ok = 'cursed' not in agent.message
        self._log(f'remove {ring.text!r}: {agent.message!r}')
        agent.inventory.items.update(force=True)
        return ok

    def _quaff(self, item):
        if jf_config.CFP_XORN and not item.is_unambiguous():
            # castle-first-pass: an unknown potion may be polymorph -- with a ring of polymorph control on, POLY_XORN
            # turns it into a xorn that walks through the walls (the kit's rings that may be control go on first:
            # amd jf16-s7/jf16-s13 carried one with no wand of polymorph)
            # (only a KNOWN ring of polymorph control: putting every unknown ring on for a ~4% potion of polymorph
            # would wear conflict, hunger, aggravation and polymorph rings through the crossing)
            from . import castle_cross
            if castle_cross.poly_prep(self, known_only=True):
                return
        self._tested.add(item.glyphs[0])
        self._lev_warn = None   # (BREACH_LEVWARN: a new lift starts its own clock)
        self.agent.inventory.quaff(item)
        self._log(f'quaffed {item.text!r}: {self.agent.message!r}')
        if self.levitating():
            self._lev_source = ('potion', item.glyphs[0])

    def _wear_boots(self, item):
        agent = self.agent
        worn = next((i for i in self._items() if i.is_armor() and i.equipped and
                     getattr(i.objs[0], 'sub', None) == getattr(item.objs[0], 'sub', None)), None)
        if worn is not None:
            if worn.status == worn.CURSED or not agent.inventory.takeoff(worn):
                self._log(f'cannot take off {worn.text!r}')
                self._tested.add(item.glyphs[0])
            return
        self._tested.add(item.glyphs[0])
        agent.inventory.wear(item)
        self._log(f'wore {item.text!r}: {agent.message!r}')
        if self.levitating():
            self._lev_source = ('boots', item.glyphs[0])

    def _wish(self, wand):
        """A wand of wishing with a charge left: a blessed ring of levitation (tried next). agent.update
        answers wish prompts itself (agent.wish_text, default gray dragon scale mail), so set it first."""
        agent = self.agent
        self._tested.add(wand.glyphs[0])
        self._log('wishing for a ring of levitation')
        old = getattr(agent, 'wish_text', None)
        agent.wish_text = 'blessed ring of levitation'
        agent.wish_purpose = 'passage'   # power.py's version of the wish hook reads this one
        try:
            with agent.atom_operation():
                agent.step(A.Command.ZAP)
                agent.type_text(agent.inventory.items.get_letter(wand))
        finally:
            agent.wish_text = old
            agent.wish_purpose = None
        self._log(f'wish: {agent.message!r}')
        if jf_config.WISH_LEARN and ('Nothing happens' in agent.message or 'You wrest' in agent.message):
            # (x:0): zappable() refuses 120 times in 121; the drill zapped it every turn (pwc-dp10 jf14-s14)
            agent.inventory.empty_wands.add(wand.text)
        agent.inventory.items.update(force=True)

    def _stop_levitating(self):
        """Come down (over the trap door this drops us through it: trap.c float_down -> dotrap)."""
        agent = self.agent
        tries = self._tries
        src = self._lev_source
        rings = [r for r in self._worn_rings() if 'levitation' in self._names(r)]
        if jf_config.CASTLE_EDGE_REST and src is not None and src[0] == 'potion':
            # a potion lifted us: a worn unknown ring is not the lift (and pulling at a cursed one wastes turns)
            rings = [r for r in rings if r.is_unambiguous()]
        if src is not None and src[0] == 'ring':
            rings = [r for r in self._worn_rings() if r.glyphs[0] == src[1]] or rings
        if rings and tries.get('remove_fail', 0) < 2:
            if not self._remove_ring(rings[0]):
                tries['remove_fail'] = tries.get('remove_fail', 0) + 1
            return
        boots = next((i for i in self._items() if i.equipped and i.is_unambiguous() and
                      i.object.name == 'levitation boots'), None)
        if boots is not None and boots.status != boots.CURSED and tries.get('boots_fail', 0) < 2:
            if not agent.inventory.takeoff(boots):
                tries['boots_fail'] = tries.get('boots_fail', 0) + 1
            return
        if (rings or boots is not None) and not agent.prayer_failed and agent.is_safe_to_pray(500) and \
                not tries.get('prayed'):
            # levitation stuck on by a cursed ring/boots is major trouble (pray.c TROUBLE_LEVITATION)
            tries['prayed'] = 1
            self._log('praying off a cursed levitation item')
            agent.pray()
            return
        # a potion: a blessed one ends at will with '>' (potion.c I_SPECIAL); otherwise wait it out
        if not tries.get('descend'):
            tries['descend'] = 1
            agent.direction('>')
            self._log(f"'>' to come down: {agent.message!r}")
            return
        agent.search(3)

    # ------------------------------------------------------------------ monsters

    def _monster_at(self, mx, my):
        y, x = to_bot(mx, my)
        if (y, x) == (self.agent.blstats.y, self.agent.blstats.x):
            return False
        glyph = self.agent.glyphs[y, x]
        return nh.glyph_is_monster(glyph) or glyph == nh.GLYPH_INVISIBLE or \
            bool(self.agent.monster_tracker.monster_mask[y, x])

    def _dry(self, mx, my):
        if map_char(mx, my) == '.':
            return True
        y, x = to_bot(mx, my)
        return self.agent.current_level().objects[y, x] == SS.S_ice or self.agent.glyphs[y, x] == SS.S_ice

    def _hostiles_near(self, radius):
        """Hostiles within radius standing on dry land (the moat's eels and sharks can't come ashore)."""
        bl = self.agent.blstats
        return [m for m in self.agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= radius and self._dry(*to_map(m[1], m[2]))]

    def _fight_adjacent(self):
        """While standing still (door work, waiting to come down) hit back at whatever is next to us on
        dry land (not the moat's eels and sharks: they submerge, and the door work is what counts)."""
        near = self._hostiles_near(1)
        if not near:
            return False
        agent = self.agent
        _, y, x, mon, _ = near[0]
        self._set_state(f'fighting {getattr(mon, "mname", "?")}')
        with agent.atom_operation():
            agent.step(A.Command.FIGHT)
            agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
        return True

    def _step_to(self, mx, my):
        """One step (or an attack on whatever blocks it). Water is fine: we float, walk on it or it is ice."""
        agent = self.agent
        y, x = to_bot(mx, my)
        d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
        if self._monster_at(mx, my):
            if jf_config.CFP_ZAP and self._floating():
                # castle-first-pass: teleportation/striking beams (or a ray along a long straight stretch) first
                from . import castle_cross
                if castle_cross.blocker_zap(self, mx, my, d):
                    return
            if jf_config.CASTLE_EDGE_REST and self._floating():
                # a blocker in the moat ring's one-square-wide columns (x 0 and 62) holds us under its bites and a
                # tower xorn's: pwc-dp5 jf14-s14 missed a xan 15 times at (62,4). A known wand of striking
                # (force bolt 2d12, never bounces back at us) first, three zaps per blocker square
                wand = next((i for i in self._items() if self._usable_wand(i, 'striking')), None)
                key = ('blocker_zap', (mx, my))
                if wand is not None and self._tries.get(key, 0) < 3:
                    self._tries[key] = self._tries.get(key, 0) + 1
                    self._set_state(f'zapping striking at what blocks {(mx, my)}')
                    agent.zap(wand, d)
                    return
            self._set_state(f'attacking what blocks {(mx, my)}')
            with agent.atom_operation():
                agent.step(A.Command.FIGHT)
                agent.direction(d)
            return
        agent.direction(d)

    def _step_downhill(self, dist, pos, prefer_route=False):
        nexts = _downhill(dist, pos)
        if not nexts:
            return False
        if jf_config.BREACH_SPOT and self.levitating() and any(self._boulder_at(*p) for p in nexts):
            # a levitating hero can't push a boulder ('You don't have enough leverage'): brx-smoke1 jf14-s14~3 tried
            # the corner's boulder 80 times and gave the passage up; go round it, or break it with the pick
            free = [p for p in nexts if not self._boulder_at(*p)]
            if not free:
                return self._smash_boulder()
            nexts = free
        route = ROUTE_S if prefer_route and self._half == 'south' else ROUTE_INDEX
        nexts.sort(key=lambda p: (self._monster_at(*p), prefer_route and p not in route))
        if jf_config.BREACH_BYPASS and self._monster_at(*nexts[0]) and self._floating():
            # every step closer is held: a free square just as close whose own next step is free goes round it -- at
            # the east channel's mouth a sea monster on (61,5) is passed by (62,5) -> (61,6) (the baseline attacked
            # 31 times from (62,4) instead; brx tally over base-all3/all4/t1-base)
            d = dist.get(pos)
            side = [(pos[0] + dx, pos[1] + dy) for dx, dy in DIRS]
            side = [q for q in side if dist.get(q) == d and not self._monster_at(*q) and
                    any(not self._monster_at(*r) for r in _downhill(dist, q))]
            if side:
                side.sort(key=lambda q: prefer_route and q not in route)
                self._set_state(f'bypassing what holds {nexts[0]} by {side[0]}')
                self._step_to(*side[0])
                return True
        self._step_to(*nexts[0])
        return True

    # ------------------------------------------------------------------ CASTLE_SEA_SWITCH

    def _note_sea(self):
        """Monsters on the moat's west half (sharks, eels, a remembered 'I'): square -> turn seen."""
        agent = self.agent
        t = agent.blstats.time
        me = (agent.blstats.y, agent.blstats.x)
        for mx in range(0, 9):
            for my in range(MAP_H):
                if map_char(mx, my) != '}':
                    continue
                y, x = to_bot(mx, my)
                if (y, x) == me:
                    continue
                g = agent.glyphs[y, x]
                if nh.glyph_is_monster(g) or g == nh.GLYPH_INVISIBLE:
                    self._sea[(mx, my)] = t

    def _pick_half(self):
        """In the west courtyard: the way round whose west end had no sea monster lately. A sea monster moves
        greedily toward us and can't round the courtyard (the moat's x 5 column is a dead end for that), so one
        that holds the north column stays there while we take the south one (pwc-dp10 jf14-s0, jf25-s11: a shark
        in the column (0,4) bit two potion-lifted crossers to death while they attacked it; jf16-s13 walked
        into a shark at (0,5) three times)."""
        t = self.agent.blstats.time

        def last(half):
            return max((turn for p, turn in self._sea.items() if west_zone(p) == half), default=None)
        cur = self._half
        other = 'south' if cur == 'north' else 'north'
        lc, lo = last(cur), last(other)
        # both held lately: stay (no dance in the courtyard: pwc-sea1 jf14-s0 turned twice in 4 turns)
        # (a sea monster hides under water 4 moves in 5 and hovers by the corner nearest us while we rest at
        # TEST_SPOT: a sighting counts for 300 turns, longer than a rest stop)
        if lc is not None and t - lc <= 300 and (lo is None or t - lo > 300):
            self._half = other
            self._log(f'sea monster in the {cur} column (turn {lc}): going round by the {other}')

    def _sea_step(self, pos, dist):
        """True: took a step backing out of a held west column. In our half's west end (or on the courtyard's
        moat edge) with every next square taken by a monster: back to the courtyard, where _pick_half turns
        us round the other way. Past the west end (the strip, the east column) we press on as before."""
        if self._sea_retreat:
            if pos in WEST_COURTYARD and pos not in MOAT_EDGE:
                self._sea_retreat = False
                self._pick_half()
                return False
            self._set_state('backing out of a held moat column')
            if not self._step_downhill(_bfs(self._tspot(), OUTSIDE), pos):
                self._sea_retreat = False
                return False
            return True
        if west_zone(pos) != self._half and pos not in MOAT_EDGE:
            return False
        nexts = _downhill(dist, pos)
        if not nexts or not all(self._monster_at(*p) for p in nexts) or self._tries.get('sea_retreat', 0) >= 4:
            return False
        self._tries['sea_retreat'] = self._tries.get('sea_retreat', 0) + 1
        self._sea_retreat = True
        self._log(f'the {self._half} column is held at {nexts}: backing out')
        self._note_sea()
        if pos in WEST_COURTYARD:
            self._sea_retreat = False
            self._pick_half()
            return False
        return self._sea_step(pos, dist)

    # ------------------------------------------------------------------ BREACH_PROBE

    def _probe_corner(self):
        """The moat-ring corner of our half (north (0,6), south (0,10)) and a dry square next to it with no water
        around (TEST_SPOT for the north corner)."""
        if self._half == 'south':
            return (0, 10), (1, 9)
        return CORNER, self._tspot()

    def _sea_adjacent(self):
        """Visible monsters on moat squares next to us (not ice): eels, sharks, a remembered 'I'."""
        mx, my = self._pos()
        out = []
        for dx, dy in DIRS:
            n = (mx + dx, my + dy)
            if map_char(*n) == '}' and not self._dry(*n) and self._monster_at(*n):
                out.append(n)
        return out

    SEA_WORDS = ('shark', 'giant eel', 'electric eel', 'kraken', 'piranha', 'jellyfish')
    WAND_SPOT = (2, 8)   # a west courtyard square with no water next to it (TEST_SPOT gets our Elbereth)
    # west courtyard squares with no water beside them, the north corner's nearest first
    INNER = ((1, 7), (2, 7), (1, 8), (2, 8), (3, 7), (1, 9), (3, 8), (2, 9), (3, 9))

    def _boulder_at(self, mx, my):
        y, x = to_bot(mx, my)
        return self.agent.glyphs[y, x] in G.BOULDER or self.agent.current_level().objects[y, x] in G.BOULDER

    def _tspot(self):
        """TEST_SPOT, unless BREACH_SPOT and a boulder lies on it (giants in the castle carry and drop boulders: the
        brx all-flags jf14-s14 explored for 4000 turns toward a (1,7) under a boulder; the baseline gave up
        'courtyard square (1, 7) not reachable' in jf14-s14~1): then the nearest free inner courtyard square."""
        if not jf_config.BREACH_SPOT or not self._boulder_at(*TEST_SPOT):
            return TEST_SPOT
        for p in self.INNER:
            if not self._boulder_at(*p):
                return p
        return TEST_SPOT

    def _wand_spot(self):
        if not jf_config.BREACH_SPOT:
            return self.WAND_SPOT
        ts = self._tspot()
        for p in (self.WAND_SPOT,) + self.INNER:
            if p != ts and not self._boulder_at(*p):
                return p
        return self.WAND_SPOT

    def _wand_test_step(self):
        """BREACH_WANDS: engrave-test every unknown wand on a bare square of the west courtyard before the passage
        plan runs out. The arrival drill's tests ran on the landing square -- in the pit our castle-detecting dig
        left, where inventory._engrave_single_wand's look ('There is a pit here. You see no objects here.') refused
        each test and marked the wand tried: pwc/brx jf14-s0 gave the castle up ('nothing that crosses water') with
        an untested wand of cold in the pack. True: acted this step."""
        agent = self.agent
        t = self._tries
        from . import castle_power
        tried = t.setdefault('bw_tried', set())
        wands = [w for w in castle_power._untested_wands(agent) if w.glyphs[0] not in tried]
        if not wands or agent.character.prop.blind or not agent.can_engrave() or \
                agent.character.prop.polymorph or t.get('bw_approach', 0) > 60:
            return False
        if any(max(abs(m[1] - agent.blstats.y), abs(m[2] - agent.blstats.x)) <= 2
               for m in agent.get_visible_monsters()):
            if t.get('bw_wait', 0) >= 30:
                return False
            t['bw_wait'] = t.get('bw_wait', 0) + 1
            agent.search()   # (fight2 deals with it; don't give the castle up meanwhile)
            return True
        spot = self._wand_spot()
        if self._pos() != spot:
            t['bw_approach'] = t.get('bw_approach', 0) + 1
            self._set_state(f'to {spot} to engrave-test {len(wands)} wands')
            return self._approach(spot)
        w = wands[0]
        tried.add(w.glyphs[0])
        self._set_state(f'engrave-testing {w.text!r}')
        try:
            ok = castle_power._engrave_test(self, w)
        except (AgentChangeStrategy, AgentFinished, AgentPanic):
            raise
        except Exception as e:   # an unexpected prompt in the engrave dialogue must not end the passage
            self._log(f'engrave test of {w.text!r} failed: {e!r}')
            ok = True
        if not ok:
            self._log(f'engrave test of {w.text!r} refused here ({agent.message[:60]!r})')
        return True

    def _probe_refresh(self):
        """BREACH_PROBE: a probe that ended more than BREACH_PROBE_STALE turns ago (potion tests that blinded or
        confused us, rests) is done again, shorter, before the next thing that may lift us."""
        t = self._tries
        now = self.agent.blstats.time
        if t.get('probe_done') and now - t.get('probe_end', now) > jf_config.BREACH_PROBE_STALE and \
                t.get('probe_round', 1) < 12:
            t['probe_round'] = t.get('probe_round', 1) + 1
            t['probe_done'] = 0
            t['probe_start'] = now
            t['probe_quiet'] = 0

    def _probe_step(self):
        """BREACH_PROBE: before the first lift is tried (or set off with), stand on the corner of our half on foot and
        search -- a search finds a hidden eel or shark next to us (detect.c mfind0) -- fighting whatever sea monster
        shows from dry land, and stepping back to rest on Elbereth one square in (no water next to it) below
        BREACH_PROBE_HP. Done after BREACH_PROBE_QUIET searches in a row with no sea monster seen or heard of, or after
        BREACH_PROBE_BUDGET turns. True: acted this step. (A shark waiting hidden at the channel entrance killed 10 of
        ~21 crossing attempts in the baseline: they found it only after the potion had lifted them over the water.)"""
        t = self._tries
        agent = self.agent
        bl = agent.blstats
        if t.get('probe_done'):
            return False
        if self.levitating() and self._timed_levitation():
            return False   # a potion's lift is running: go
        first = t.get('probe_round', 1) == 1
        quiet_needed = jf_config.BREACH_PROBE_QUIET if first else max(3, jf_config.BREACH_PROBE_QUIET // 2)
        budget = jf_config.BREACH_PROBE_BUDGET if first else jf_config.BREACH_PROBE_BUDGET // 3
        now = bl.time
        start = t.setdefault('probe_start', now)
        if now - start > budget:
            t['probe_done'] = 1
            t['probe_end'] = now
            self._log(f'probe: budget of {budget} turns spent')
            return False
        msg = (agent.message or '').lower()
        if any(w in msg for w in self.SEA_WORDS):
            t['probe_quiet'] = 0   # bitten, found, fled: a sea monster is about even if hidden again now
        corner, inner = self._probe_corner()
        pos = self._pos()
        resting = t.get('probe_resting')
        if bl.hitpoints < jf_config.BREACH_PROBE_HP * bl.max_hitpoints or \
                (resting and bl.hitpoints < 0.9 * bl.max_hitpoints):
            if not resting:
                self._log(f'probe: resting at {inner}, hp {bl.hitpoints}/{bl.max_hitpoints}')
            t['probe_resting'] = 1
            if pos != inner:
                self._set_state(f'probe: stepping back to {inner}')
                if not self._step_downhill(_bfs(inner, OUTSIDE), pos):
                    return self._approach(inner)
                return True
            if self.levitating():
                self._stop_levitating()   # a ring or boots: back on the floor to engrave and rest
                return True
            engraving = (agent.inventory.engraving_below_me or '').lower()
            if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind:
                agent.engrave('Elbereth')
                return True
            if not self._fight_adjacent():
                self._set_state('probe: resting')
                agent.search(3)
            return True
        t['probe_resting'] = 0
        if pos != corner:
            if max(abs(pos[0] - corner[0]), abs(pos[1] - corner[1])) == 1:
                self._set_state(f'probe: stepping onto the corner {corner}')
                self._step_to(*corner)
                return True
            return self._approach(inner)
        sea = self._sea_adjacent()
        if sea:
            t['probe_quiet'] = 0
            t['probe_seen'] = t.get('probe_seen', 0) + 1
            self._set_state(f'probe: fighting the sea monster at {sea[0]}')
            y, x = to_bot(*sea[0])
            with agent.atom_operation():
                agent.step(A.Command.FIGHT)
                agent.direction(agent.calc_direction(bl.y, bl.x, y, x))
            return True
        if self._fight_adjacent():
            return True
        t['probe_quiet'] = t.get('probe_quiet', 0) + 1
        if t['probe_quiet'] > quiet_needed:
            t['probe_done'] = 1
            t['probe_end'] = now
            self._log(f'probe {t.get("probe_round", 1)}: {corner} quiet for {quiet_needed} searches '
                      f'(sea monsters met: {t.get("probe_seen", 0)}, {now - start} turns): setting off')
            return False
        self._set_state(f'probe: searching on {corner}')
        agent.search()
        return True

    # ------------------------------------------------------------------ strategy entry points

    def committed(self):
        """Cheap test (runs after every step): on the way across, where retreat_upstairs, elbereth_rest
        and fight2 would only drag us back to land or up the stairs and waste the levitation."""
        if not self.active():
            return False
        pos = self._pos()
        if pos in (DOOR, TRAPDOOR):
            return True
        if pos not in OUTSIDE:
            # still in the west maze, even if already floating: fights and rests stay with the usual layers
            # (castle-c1 scenarios: floating through the maze under this layer, 8 of 10 died there unfought)
            return False
        return pos not in WEST_COURTYARD or self._floating()

    def crossing_strategy(self):
        """Keeps control for the whole crossing: after a single step the lower layers (elbereth_rest,
        fight2) got one action in -- jf27 seed 0 floated off on a potion and tried to engrave on the water.
        Only the emergency layer (prayer, healing potions) can still interrupt."""
        def f():
            if not self.committed():
                yield False
            yield True
            steps = 0
            while self.committed() and steps < 400:
                before = self.agent.step_count
                self._cross_step()
                steps += 1
                if self.agent.step_count == before:
                    self.agent.search()

        return Strategy(f)

    def _wield_weapon(self):
        """The failed dig leaves the pick-axe in hand (jf27 seed 0 bashed a shark with it): the crossing's
        fights want the real weapon. Once per castle visit."""
        if self._tries.get('wielded'):
            return False
        self._tries['wielded'] = 1
        return bool(self.agent.wield_best_melee_weapon())

    def _allow_traps(self):
        """The pit our failed dig left behind usually sits in a 1-wide maze corridor, and AutoAscend's BFS
        walls every known trap off: 2 of the first 10 castle-mode replays sat thousands of turns in a
        dead-end pocket behind their own pit (STALL reachable=1). A pit only costs a few turns to cross."""
        agent = self.agent
        if agent._last_turn - agent._allow_walking_through_traps_turn > 40:
            agent._allow_walking_through_traps_turn = agent._last_turn
            agent.last_bfs_step = -1   # the BFS cache ignores the flag

    def _guard_moat_edge(self):
        """Once inside the courtyard, keep the BFS (and fight2's steps, which follow it) off the squares
        beside the moat. The maze's way in, (0,10), is one of them, so not before we are in."""
        if self._tries.get('edge_marked') or self._pos() not in WEST_COURTYARD:
            return
        self._tries['edge_marked'] = 1
        level = self.agent.current_level()
        for p in MOAT_EDGE:
            if p != self._pos():
                level.forbidden[to_bot(*p)] = True
        self.agent.last_bfs_step = -1

    def _approach(self, spot):
        """Walk (exploring the west maze if need be) to a courtyard square."""
        agent = self.agent
        self._allow_traps()
        self._guard_moat_edge()
        target = to_bot(*spot)
        if spot in MOAT_EDGE and agent.bfs()[target] == -1:
            # the corner is off limits to the BFS: step onto it from a neighbour by hand
            here = self._pos()
            if max(abs(here[0] - spot[0]), abs(here[1] - spot[1])) == 1:
                self._set_state(f'stepping to {spot}')
                self._step_to(*spot)
                return True
            spot = self._tspot()
            target = to_bot(*spot)
        if agent.bfs()[target] == -1:
            mx, my = self._pos()
            if jf_config.CASTLE_WEST_DIG and mx < 0 and self.dive.digging_tool() is not None:
                # dig straight east into the courtyard instead of exploring the dark west maze for its single
                # join, map (-1,10) (castle.des MAZEWALK:(00,10),west): every maze wall outside the map box is
                # diggable and the courtyard's west column (map x 0, rows 6-10) is floor. Rows 7-9 keep the
                # entry square off the moat. The walk through the maze is where the minotaurs meet us
                # (pwc-dp1: 20 of 61 real kits killed by one, mostly before reaching the courtyard).
                row = max(7, min(9, int(my)))
                if self._dig_toward((0, row)):
                    return True
            self._approach_tries += 1
            if self._approach_tries > 60:
                self._give_up(f'courtyard square {spot} not reachable')
                return False
            # boulders (fill_empty_maze puts 1-5 in each maze half) plug the 1-wide corridors, and the BFS
            # never pushes one: jf16 seed 6 sat 5000 turns between two of them (STALL reachable=1)
            if self._approach_tries > 1 and self._smash_boulder():
                return True
            self._set_state('exploring to the courtyard')
            start = agent.step_count
            self.dive.exploration(None).until(
                agent, lambda: agent.bfs()[target] != -1 or agent.step_count - start > 300).run()
            return True
        self._set_state(f'walking to {spot}')
        agent.go_to(*target)
        return True

    EAST_JOIN = (63, 6)   # castle.des MAZEWALK:(62,06),east: the east maze meets the courtyard only here

    def _wet_around(self, mx, my):
        """Moat (or known water) next to map square (mx, my): sea monsters reach it (eel wraps drown, F030)."""
        level = self.agent.current_level()
        for dx, dy in DIRS:
            n = (mx + dx, my + dy)
            if map_char(*n) == '}' and not self._dry(*n):
                return True
            y, x = to_bot(*n)
            if 0 <= y < level.objects.shape[0] and 0 <= x < level.objects.shape[1] and \
                    level.objects[y, x] in (SS.S_pool, SS.S_water):
                return True
        return False

    def _east_step(self):
        """Arrived on the castle's east side (up the Valley's '<': levregion 69-79 = the courtyard's east column
        or the east maze). Heal first after a Valley retreat -- on a square with no water beside it (gehennom's
        retreats were drowned by moat eels while resting at the courtyard edge), else at SAFE_EAST -- then walk to
        the back door, or, when the maze hides the way, dig straight to the one square where it joins the
        courtyard, map (63,6): exploring the maze took gehennom's retreats 100-1600 turns, most dying there."""
        agent = self.agent
        bl = agent.blstats
        tries = self._tries
        pos = self._pos()
        self._allow_traps()
        heal = self._after_valley_retreat() and bl.hitpoints < 0.9 * bl.max_hitpoints and tries.get('east_rest', 0) < 600
        if heal and not self._wet_around(*pos):
            tries['east_rest'] = tries.get('east_rest', 0) + 1
            near = [m for m in self._hostiles_near(1) if self.dive._ignores_elbereth(m[3])]
            if near:
                return self._fight_adjacent() or bool(agent.search())
            engraving = (agent.inventory.engraving_below_me or '').lower()
            if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind:
                self._set_state('east side: Elbereth to heal after the Valley')
                agent.engrave('Elbereth')
                return True
            self._set_state('east side: healing after the Valley')
            agent.search(3)
            return True
        target = to_bot(*(SAFE_EAST if heal else GOAL))
        if agent.bfs()[target] != -1:
            self._set_state('east side: walking to ' + ('SAFE_EAST to heal' if heal else 'the back door'))
            agent.go_to(*target)
            return True
        if self._dig_toward(self.EAST_JOIN, then=(self.EAST_JOIN[0] - 1, self.EAST_JOIN[1])):
            return True
        self._set_state('east side: exploring to the courtyard')
        self.dive.exploration(None).until(agent, lambda: agent.bfs()[to_bot(*GOAL)] != -1).run()
        return True

    def _dig_toward(self, goal, then=None):
        """One step (or one dig) on a straight path to goal: rows first, then columns. At the goal step onto
        `then` if given; False when there is nothing (more) to do here. Walls and rock of the mazes beside the
        castle map are diggable (NON_DIGGABLE covers only the map), a wall takes a dwarf 2-3 turns."""
        agent = self.agent
        mx, my = self._pos()
        side = 'east' if mx >= 57 else 'west'
        if (mx, my) == goal:
            if then is None:
                return False
            self._step_to(*then)
            return True
        if my != goal[1]:
            n = (mx, my + (1 if goal[1] > my else -1))
        else:
            n = (mx + (1 if goal[0] > mx else -1), my)
        y, x = to_bot(*n)
        if self._monster_at(*n):
            self._step_to(*n)   # attacks it
            return True
        level = agent.current_level()
        if level.walkable[y, x] and agent.glyphs[y, x] not in G.BOULDER:
            self._set_state(f'{side} side: walking toward {goal}')
            self._step_to(*n)
            return True
        key = ('dig_to', n)
        if self._tries.get(key, 0) >= 4:
            self._give_up(f'could not dig through to {n} ({side} side)')
            return False
        self._tries[key] = self._tries.get(key, 0) + 1
        tool = self.dive.digging_tool()
        if tool is None:
            return False
        d = agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x)
        self._set_state(f'{side} side: digging {d} toward {goal}')
        with agent.atom_operation():
            tool = agent.inventory.move_to_inventory(tool)
            agent.step(A.Command.APPLY)
            agent.type_text(agent.inventory.items.get_letter(tool))
            if 'In what direction do you want to dig?' in agent.single_message:
                agent.direction(d)
            elif agent.single_message.startswith('In what direction'):
                agent.step(A.Command.ESC)
        self._log(f'{side} dig {d} at {n}: {agent.message!r}')
        if 'through thin air' in agent.message:
            # nothing to dig: the square is open (the map memory hadn't marked it walkable) -- step in next time
            # (pwc-dp3 jf27-s0 swung at the air three times, gave the dig up and explored into a retreat)
            level.walkable[y, x] = True
            self._tries[key] = 0
            agent.last_bfs_step = -1
        self._tries.pop('wielded', None)   # the pick-axe is in hand now
        return True

    def _smash_boulder(self):
        """Dig through a boulder next to us with the pick-axe ('The boulder falls apart'), at most three
        tries per square."""
        agent = self.agent
        tool = self.dive.digging_tool()
        if tool is None:
            return False
        y0, x0 = agent.blstats.y, agent.blstats.x
        for y, x in agent.neighbors(y0, x0, shuffle=False):
            if agent.glyphs[y, x] not in G.BOULDER:
                continue
            key = ('boulder', int(y), int(x))
            if self._tries.get(key, 0) >= 3:
                continue
            self._tries[key] = self._tries.get(key, 0) + 1
            d = agent.calc_direction(y0, x0, y, x)
            self._set_state(f'digging through a boulder {d}')
            with agent.atom_operation():
                tool = agent.inventory.move_to_inventory(tool)
                agent.step(A.Command.APPLY)
                agent.type_text(agent.inventory.items.get_letter(tool))
                if 'In what direction do you want to dig?' in agent.single_message:
                    agent.direction(d)
                elif agent.single_message.startswith('In what direction'):
                    agent.step(A.Command.ESC)
            self._log(f'boulder dig {d}: {agent.message!r}')
            self._tries.pop('wielded', None)   # the pick-axe is in hand again
            return True
        return False

    def plan_step(self):
        """Dive plan hook on the castle level while still on the west side: walk to the courtyard, then
        freeze the moat or try what might lift us over it. False: nothing (left) to try."""
        if not self.active():
            return False
        agent = self.agent
        if not self._arrival_logged:
            self._arrival_logged = True
            inv = '; '.join(i.text for i in self._items())
            self._log(f'arrival pos={self._pos()} plan={[k for k, _ in self._plan()]} '
                      f'cold={self._cold_source() is not None} inv: {inv}')
        if self.committed():
            self._cross_step()
            return True
        if self._pos()[0] >= 57:
            # east of the castle (the east maze: back up the Valley's '<'): the back door is right here
            return self._east_step()
        if self._power_hook('arrival_step', False):
            return True
        if self._floating():
            # lifted already (a ring put on where we landed) but still in the west maze: to the corner
            return self._approach(CORNER)
        plan = self._plan()
        cold = self._cold_source()
        if jf_config.BREACH_WANDS and self._wand_test_step():
            return True
        if not plan and cold is None:
            self._give_up('nothing that crosses water')
            return False
        pos = self._pos()
        self._guard_moat_edge()
        if jf_config.CASTLE_SEA_SWITCH and pos in WEST_COURTYARD:
            self._note_sea()
        if pos in WEST_COURTYARD and self._wield_weapon():
            return True
        # Everything is tried one square inside the courtyard (a levitating hero in the maze throws daggers
        # and hurtles backwards into walls -- castle-c4 seed 1 'killed by bumping into a wall'): lasting
        # lifts first (known lev ring/boots, a wish), then untested rings/boots/amulets, then potions.
        # A cold ray can't carry us: each frozen moat square costs the ray 3 extra range (zap.c
        # zap_over_floor), so a zap freezes 2-3 squares of the 26 on the way round -- it only helps a
        # stranded crossing.
        kind, item = plan[0] if plan else ('cold', cold)
        if jf_config.BREACH_COLD_FIRST and cold is not None and kind in ('potion', 'amulet'):
            # a known cold source before the untested potions: a ray freezes ~3 moat squares, so 4-5 zaps bridge the
            # west channel (and its ice has no water beside it: no sea monster reaches us there); the potions are
            # then tried at the strip's east end (_cross_step 'stranded'), where a lift only has to last the 13 squares
            # of the east channel instead of the whole 75-square way round (an uncursed potion lasts 11-150 turns)
            kind, item = 'cold', cold
        if jf_config.BREACH_PROBE and kind not in ('horn', 'cold') and pos in WEST_COURTYARD:
            self._probe_refresh()
            if self._probe_step():
                return True
        spot = CORNER if kind in ('horn', 'cold') else self._tspot()
        if pos != spot:
            return self._approach(spot)
        if kind == 'cold':
            if self._walk_step(pos):
                return True
            self._tries['cold_done'] = 1
            self._log('cold ray: no progress from the corner')
            agent.search()
            return True
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if spot == self._tspot() and engraving != 'elbereth' and agent.can_engrave() and \
                not agent.character.prop.blind and not self._tries.get('spot_elbereth'):
            self._tries['spot_elbereth'] = 1
            agent.engrave('Elbereth')
            return True
        lasting = kind in ('wish', 'boots', 'ring', 'amulet')
        bl = agent.blstats
        if lasting and bl.hitpoints < (self._rush_hp() if jf_config.BREACH_RUSH else 0.85) * bl.max_hitpoints and \
                not self._hostiles_near(1) and self._tries.get('rest', 0) < 300:
            # a lasting lift gives time: start the gauntlet (2 sharks, 4 giant eels, xorns in the walls)
            # at full strength
            self._tries['rest'] = self._tries.get('rest', 0) + 1
            self._set_state('resting before the crossing')
            agent.search(5)
            return True
        if kind == 'potion' and self._hostiles_near(6) and self._tries.get('potion_wait', 0) < 20:
            # a potion may put us to sleep or paralyse us: not with a hostile about (fight2 deals with it)
            self._tries['potion_wait'] = self._tries.get('potion_wait', 0) + 1
            agent.search()
            return True
        self._try(kind, item)
        return True

    # ------------------------------------------------------------------ crossing

    def _walk_step(self, pos):
        """Without levitation: follow the route over dry land and ice, freezing the water ahead with a
        cold ray (straight segments, so one zap covers a whole stretch). False: stuck here."""
        i = ROUTE_INDEX.get(pos)
        if i is not None:
            if i + 1 >= len(ROUTE):
                return False
            nxt = ROUTE[i + 1]
            if self._dry(*nxt):
                self._set_state('walking the ice' if map_char(*nxt) == '}' else 'walking across')
                self._step_to(*nxt)
                return True
            cold = self._cold_source()
            return cold is not None and self._cold_step(cold)
        # off the route: walk to the most advanced route square we can reach dry-shod
        dry = frozenset(p for p in OUTSIDE if self._dry(*p))
        if pos not in dry:
            return False
        reach = _bfs(pos, dry)
        on_route = [(ROUTE_INDEX[p], p) for p in reach if p in ROUTE_INDEX]
        if not on_route:
            return False
        target = max(on_route)[1]
        self._set_state(f'walking to the route at {target}')
        return self._step_downhill(_bfs(target, dry), pos)

    def _after_valley_retreat(self):
        """gehennom climbs back up the Valley's '<' when swarmed (dive.valley_retreat_turn): heal before
        the trap door drops us back in."""
        turn = getattr(self.dive, 'valley_retreat_turn', None)
        return turn is not None and self.agent.blstats.time - turn < 1000

    def _rest_stop(self, pos):
        """A dry square on the way (the strip along row 0, a courtyard) with the lift under our control
        (ring/boots, or none after a Valley retreat): take it off, write Elbereth (eels, sharks and xorns
        respect it) and rest before going on. Timed levitation (a potion) can't wait."""
        agent = self.agent
        bl = agent.blstats
        if map_char(*pos) != '.' or pos in (DOOR, TRAPDOOR):
            self._resting = False
            return False
        if self.levitating() and self._timed_levitation():
            self._resting = False
            return False
        if jf_config.CFP_MB and not self.levitating():
            # castle-first-pass: a polymorph form doesn't rest -- its hit points are its own pool (rehumanize returns
            # ours) and it ends in rn1(500,500) turns; a cursed ring of polymorph may change it any turn (power-route's
            # arm jf25 s1: a human mummy rested ~125 turns, then a panther and a shark killed the form)
            from . import castle_cross
            if castle_cross.form_permonst(agent) is not None and castle_cross.amphibious(self):
                self._resting = False
                return False
        want = 0.9 if (self._resting or self._after_valley_retreat()) else 0.45
        if jf_config.BREACH_RUSH and not self._resting and map_char(*pos) == '.' and pos[1] in (0, 16):
            # BREACH_RUSH: the strip is where a lasting lift rests (no land monster reaches it, Elbereth holds the sea
            # monsters in row 1): stop there below 80% instead of in the west courtyard
            want = 0.8
        if bl.hitpoints >= want * bl.max_hitpoints or self._tries.get('rest_stop', 0) > 600:
            if self._resting:
                self._log(f'rested: hp {bl.hitpoints}/{bl.max_hitpoints}')
            self._resting = False
            return False
        if jf_config.CASTLE_EDGE_REST and (pos in MOAT_EDGE or (pos[0] >= 57 and self._wet_around(*pos))):
            # rest one square in from the moat (sharks, eel wraps), not on its edge: TEST_SPOT in the west
            # courtyard, SAFE_EAST in the east one (pwc-dp4 jf25-s0 rested at the east courtyard's edge (61,6))
            spot = SAFE_EAST if pos[0] >= 57 else self._tspot()
            self._set_state(f'stepping off the moat edge to rest at {spot}')
            if not self._step_downhill(_bfs(spot, OUTSIDE), pos):
                self._step_to(*spot)
            return True
        if not self._resting:
            self._log(f'rest stop at {pos}: hp {bl.hitpoints}/{bl.max_hitpoints}')
        self._resting = True
        self._tries['rest_stop'] = self._tries.get('rest_stop', 0) + 1
        if self.levitating():
            self._stop_levitating()   # on land: comes down where we stand
            return True
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind:
            agent.engrave('Elbereth')
            return True
        # Elbereth doesn't stop @ (soldiers) or minotaurs: hit them back
        near = [m for m in self._hostiles_near(1) if self.dive._ignores_elbereth(m[3])]
        if near:
            _, y, x, mon, _ = near[0]
            with agent.atom_operation():
                agent.step(A.Command.FIGHT)
                agent.direction(agent.calc_direction(bl.y, bl.x, y, x))
            return True
        agent.search(3)
        return True

    def _attack_holder(self):
        """A giant eel wrapped round us drowns us with its next touch (mhitu.c AD_WRAP; levitation doesn't
        help), and walking away escapes only 1 time in 13: hit it instead (jf27 seed 0: drowned after
        three 'You cannot escape from the giant eel!')."""
        agent = self.agent
        y0, x0 = agent.blstats.y, agent.blstats.x
        for y, x in agent.neighbors(y0, x0, shuffle=False):
            mx, my = to_map(y, x)
            glyph = agent.glyphs[y, x]
            if map_char(mx, my) == '}' and (nh.glyph_is_monster(glyph) or glyph == nh.GLYPH_INVISIBLE):
                self._set_state(f'held: attacking {(mx, my)}')
                with agent.atom_operation():
                    agent.step(A.Command.FIGHT)
                    agent.direction(agent.calc_direction(y0, x0, y, x))
                return True
        return False

    def _cross_step(self):
        agent = self.agent
        pos = self._pos()
        if pos == self._last_pos and not (jf_config.CASTLE_EDGE_REST and (self._resting or self._waiting)):
            self._stuck += 1
        else:
            # (a deliberate rest stop or a wait for a potion to end is not 'no progress': pwc-dp4 jf25-s0 floated
            # all the way to the east courtyard, rested there at 23 HP and was given up after 80 resting steps)
            self._stuck = 0
        self._last_pos = pos
        self._log(f'x {pos} hp {agent.blstats.hitpoints} lev {int(self.levitating())} t {agent.blstats.time}')
        if jf_config.CASTLE_SEA_SWITCH:
            self._note_sea()   # every step: a shark shows only on the turn it bites (then the rest stop takes over)
        msg = agent.message
        if 'swings itself around you' in msg or 'cannot escape from' in msg:
            self._held_turn = agent.blstats.time
        if self._held_turn is not None and agent.blstats.time - self._held_turn <= 3 and self._attack_holder():
            return
        if self._stuck > 80:
            self._give_up(f'no progress at {pos}')
            agent.search()
            return
        if self._waiting and not self.levitating():
            self._waiting = False
            self._log('levitation over: back to the door')
        if self._waiting:
            # a potion's levitation with nothing to open the door: wait for it to end out of the eels' reach
            if pos != SAFE_EAST and self._step_downhill(_bfs(SAFE_EAST, OUTSIDE), pos):
                return
            if not self._fight_adjacent():
                self._set_state('waiting for the levitation to end')
                agent.search(3)
            return
        if jf_config.BREACH_LEVWARN and self.levitating() and self._timed_levitation() and self._levwarn_step(pos):
            return
        if self._rest_stop(pos):
            return
        if jf_config.CFP_ZAP:
            # castle-first-pass: open the back door from the east courtyard's row 8, out of the eels' reach
            from . import castle_cross
            if castle_cross.door_zap_from_afar(self, pos):
                return
        if pos in (GOAL, DOOR, TRAPDOOR):
            self._door_step(pos)
            return
        if self._floating():
            if self._wield_weapon():
                return
            bl = agent.blstats
            if jf_config.BREACH_PROBE and pos in WEST_COURTYARD and not self._timed_levitation() and \
                    self._probe_step():
                return
            form_lift = False
            if jf_config.CFP_MB and not self.levitating():
                from . import castle_cross
                form_lift = castle_cross.form_permonst(agent) is not None   # (no rest in a form: see _rest_stop)
            if pos in WEST_COURTYARD and not self._timed_levitation() and not form_lift and \
                    bl.hitpoints < (self._rush_hp() if jf_config.BREACH_RUSH else 0.85) * bl.max_hitpoints and \
                    self._tries.get('rest', 0) < 300:
                # floating on a ring/boots in the courtyard: set off at full strength (fight what comes)
                if jf_config.CASTLE_EDGE_REST and pos in MOAT_EDGE:
                    # not beside the moat: a shark bit a ring-lifted pwc-dp3 jf16-s13 from 80 to 19 HP in two
                    # turns while it 'rested' on the corner (0,6); one step in there is none
                    self._set_state('stepping off the moat edge to rest')
                    # (the south corner (0,10) is 3 steps from TEST_SPOT: a direct step asserted in a loop,
                    # pwc-dp11 jf14-s14)
                    if not self._step_downhill(_bfs(self._tspot(), OUTSIDE), pos):
                        self._step_to(*self._tspot())
                    return
                self._tries['rest'] = self._tries.get('rest', 0) + 1
                if not self._fight_adjacent():
                    self._set_state('resting before the crossing')
                    agent.search(3)
                return
            # one fixed way round (the north half), pressing on through whatever blocks it: sharks bite 5d6
            # and follow along the ring, so turning back only doubles the time in their reach
            dist = _bfs(GOAL, NORTH)
            if jf_config.CASTLE_SEA_SWITCH:
                # ... except at the ring's west end: a sea monster holding one half's column is left there and
                # the other half taken (the choice is made only in the courtyard: no dance in the channel)
                self._note_sea()
                if pos in WEST_COURTYARD and not self._sea_retreat:
                    self._pick_half()
                dist = _bfs(GOAL, SOUTH if self._half == 'south' else NORTH)
                if pos in dist and self._sea_step(pos, dist):
                    return
                dist = _bfs(GOAL, SOUTH if self._half == 'south' else NORTH)
            if pos not in dist:
                dist = _bfs(GOAL, OUTSIDE)
            if jf_config.BREACH_SIDESTEP and self._sidestep(pos):
                return
            if pos in dist:
                self._set_state('floating across')
                if self._step_downhill(dist, pos, prefer_route=True):
                    return
            elif pos not in OUTSIDE:
                # floating somewhere off the castle map (the west maze): get to the courtyard first
                corner = to_bot(*CORNER)
                if agent.bfs()[corner] != -1:
                    self._set_state('floating to the courtyard')
                    agent.go_to(*corner, max_steps=1)
                    return
                self._set_state('exploring to the courtyard (floating)')
                self.dive.exploration(None).until(agent, lambda: agent.bfs()[corner] != -1).run()
                return
        elif self._walk_step(pos):
            return
        # stranded (levitation ended on the dry strip, the ice ran out): try whatever is left right here
        plan = self._plan()
        if plan:
            if jf_config.BREACH_COLD_FIRST and not self.levitating() and map_char(*pos) == '.':
                # the strip borders the moat (row 1) all along: Elbereth first, so a quaff that puts us to sleep or
                # blinds us doesn't hand a shark free bites (sea monsters respect it)
                engraving = (agent.inventory.engraving_below_me or '').lower()
                if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind and \
                        self._tries.get('strand_elbereth', 0) < 30:
                    self._tries['strand_elbereth'] = self._tries.get('strand_elbereth', 0) + 1
                    self._set_state(f'stranded at {pos}: Elbereth before trying {plan[0][0]}')
                    agent.engrave('Elbereth')
                    return
            self._set_state(f'stranded at {pos}')
            self._try(*plan[0])
            return
        self._give_up(f'stranded at {pos}')
        agent.search()

    # the west channels' mouths: the corner (dry), the two water squares off it, and the first square of the
    # one-square-wide column that both of them lead into
    MOUTHS = {'north': ((0, 6), ((0, 5), (1, 5)), (0, 4)), 'south': ((0, 10), ((0, 11), (1, 11)), (0, 12))}

    def _sidestep(self, pos):
        """BREACH_SIDESTEP: a sea monster in the first square of the one-square-wide west column (the harness baseline's
        commonest block: 42 attack actions at (0,4) from (0,5), each answered by 5d6 shark bites and a xorn in the walls)
        can't be passed; one on either of the two squares off the corner can -- both lead diagonally into the column.
        So back off onto the corner and wait there (up to 4 turns, twice) for the blocker to come out after us, then go
        round it; the downhill step already prefers the unoccupied mouth square. True: acted this step."""
        agent = self.agent
        t = self._tries
        corner, mouth, head = self.MOUTHS['south' if self._half == 'south' else 'north']
        now = agent.blstats.time
        waiting = t.get('ss_wait_since')
        if pos in mouth and self._monster_at(*head) and t.get('ss_rounds', 0) < 2 and not self._monster_at(*corner):
            t['ss_rounds'] = t.get('ss_rounds', 0) + 1
            t['ss_wait_since'] = now
            self._log(f'sidestep {t["ss_rounds"]}: {head} is held -- back onto {corner} to draw it out')
            self._set_state('sidestep: backing onto the corner')
            self._step_to(*corner)
            return True
        if pos == corner and waiting is not None:
            if now - waiting > 4 or not self._monster_at(*head) or any(self._monster_at(*m) for m in mouth):
                t['ss_wait_since'] = None   # it came out (or left): go on, round it
                return False
            self._set_state('sidestep: waiting on the corner for the blocker to come out')
            if not self._attack_holder():
                agent.search()
            return True
        return False

    def _rush_hp(self):
        """BREACH_RUSH: the HP fraction below which a lasting lift still rests in the west courtyard before the crossing.
        The courtyard is where the maze's land monsters reach us (brx-t2-mino: ring kits 4/24, most deaths at the
        landing or during the courtyard rest -- jf16-s6~2 rested floating at 38-50% HP until a gnome king killed it);
        two squares into the channel no land monster reaches us, and the strip (row 0) is the rest stop (Elbereth
        holds the sea monsters). A land hostile within 4: no rest at all -- the moat is the escape."""
        agent = self.agent
        bl = agent.blstats
        y0, x0 = bl.y, bl.x
        land = [m for m in agent.get_visible_monsters()
                if max(abs(m[1] - y0), abs(m[2] - x0)) <= 4 and
                (map_char(*to_map(m[1], m[2])) != '}' or self._dry(*to_map(m[1], m[2])))]   # (the maze counts)
        if land:
            return 0.0
        return jf_config.BREACH_RUSH_HP

    def _levwarn_step(self, pos):
        """BREACH_LEVWARN: timeout.c levitation_dialogue says 'You float slightly lower.' 5 turns before a potion's
        lift ends ('You wobble unsteadily' at 3). Then: quaff another known potion of levitation (potion.c adds to
        the timeout even while levitating); else don't start a water stretch from dry land, and over water make for
        the nearest dry square within reach instead of pressing on (a lift ending over the one-square channels
        drowns us: no land to crawl out to). True: acted this step."""
        agent = self.agent
        msg = agent.message or ''
        now = agent.blstats.time
        if 'float slightly lower' in msg or 'wobble unsteadily' in msg:
            if getattr(self, '_lev_warn', None) is None:
                self._log(f'levitation ending (turn {now}) at {pos}')
            self._lev_warn = now
        warn = getattr(self, '_lev_warn', None)
        if warn is None or now - warn > 8:
            return False
        pot = next((i for i in self._items() if i.category == nh.POTION_CLASS and i.is_unambiguous() and
                    i.object.name == 'levitation' and i.status != i.CURSED), None)
        if pot is not None:
            self._lev_warn = None
            self._set_state(f'levitation ending: quaffing {pot.text!r} to stay up')
            agent.inventory.quaff(pot)
            self._log(f'quaffed {pot.text!r} over {pos}: {agent.message!r}')
            return True
        dist_goal = _bfs(GOAL, OUTSIDE)
        if self._dry(*pos):
            nexts = _downhill(dist_goal, pos)
            if nexts and all(not self._dry(*n) for n in nexts):
                self._set_state('levitation ending: waiting on dry land')
                if not self._fight_adjacent():
                    agent.search()
                return True
            return False
        reach = _bfs(pos, OUTSIDE)
        land = [(d, dist_goal.get(p, 999), p) for p, d in reach.items() if d <= 6 and self._dry(*p)]
        if not land:
            return False
        _, _, target = min(land)
        self._set_state(f'levitation ending: making for dry land at {target}')
        if not self._step_downhill(_bfs(target, OUTSIDE), pos):
            return False
        return True

    def _cold_step(self, cold):
        """Standing on the route with water next: freeze the straight segment ahead with a cold ray."""
        agent = self.agent
        pos = self._pos()
        i = ROUTE_INDEX.get(pos)
        if i is None or i + 1 >= len(ROUTE):
            return False
        nxt = ROUTE[i + 1]
        if self._dry(*nxt):
            return False
        key = ('cold', pos)
        if self._tries.get(key, 0) >= 3:
            self._tries['cold_done'] = 1
            return False
        self._tries[key] = self._tries.get(key, 0) + 1
        d = agent.calc_direction(agent.blstats.y, agent.blstats.x, *to_bot(*nxt))
        self._set_state(f'freezing the moat {d} of {pos}')
        if cold.category == nh.WAND_CLASS:
            agent.zap(cold, d)
        else:
            with agent.atom_operation():
                agent.step(A.Command.APPLY)
                agent.type_text(agent.inventory.items.get_letter(cold))
                if 'In what direction?' in agent.single_message:
                    agent.direction(d)
        self._log(f'cold ray {d}: {agent.message!r}')
        return True

    # ------------------------------------------------------------------ the back door

    def _door_open(self):
        y, x = to_bot(*DOOR)
        glyph = self.agent.glyphs[y, x]
        if glyph in G.DOOR_CLOSED:
            return False
        if glyph in G.DOOR_OPENED or glyph in (SS.S_ndoor, SS.S_room, SS.S_darkroom):
            return True
        # a monster or an object in the doorway hides it: go by what we remember
        return self.agent.current_level().objects[y, x] not in G.DOOR_CLOSED

    def _door_step(self, pos):
        if self._power_hook('door_step', False, pos): return
        agent = self.agent
        tries = self._tries
        if pos == TRAPDOOR:
            if self.levitating():
                self._set_state('over the trap door: coming down')
                self._stop_levitating()
                return
            # 'You escape a trap door.': step off and on again
            self._set_state('trap door did not trigger: stepping off')
            self._step_to(*DOOR)
            return
        if pos == DOOR:
            self._set_state('stepping onto the trap door')
            self._step_to(*TRAPDOOR)
            return
        # in front of the door
        if self._door_open():
            self._set_state('through the door')
            self._step_to(*DOOR)
            return
        if jf_config.BREACH_DOOR and not self.levitating():
            # the east eels start at (57,07) and (57,09), both next to this square: Elbereth under us while we work
            # the lock (a scared eel lets go, F030); a kick wipes it (dokick.c u_wipe_engr) -- written again after
            engraving = (agent.inventory.engraving_below_me or '').lower()
            if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind and \
                    tries.get('door_elbereth', 0) < 80:
                tries['door_elbereth'] = tries.get('door_elbereth', 0) + 1
                self._set_state('Elbereth at the back door')
                agent.engrave('Elbereth')
                return
        for name in ('striking', 'digging', 'opening'):
            wand = next((i for i in self._items() if self._usable_wand(i, name)), None)
            if wand is not None and tries.get(name, 0) < 2:
                tries[name] = tries.get(name, 0) + 1
                self._set_state(f'zapping {name} at the door')
                agent.zap(wand, 'w')
                self._log(f'zap {name}: {agent.message!r}')
                return
        if tries.get('open', 0) < 2:
            # autoopen: walking into a closed door opens it unless it is locked
            tries['open'] = tries.get('open', 0) + 1
            self._set_state('opening the door')
            agent.direction('w')
            self._log(f'open: {agent.message!r}')
            return
        tool = next((i for i in self._items() if i.is_unambiguous() and
                     i.object.name in ('skeleton key', 'lock pick', 'credit card')), None)
        if tool is not None and tries.get('unlock', 0) < 8:
            tries['unlock'] = tries.get('unlock', 0) + 1
            tries['open'] = 0
            self._set_state(f'unlocking the door with {tool.text!r}')
            with agent.atom_operation():
                agent.step(A.Command.APPLY)
                agent.type_text(agent.inventory.items.get_letter(tool))
                if 'direction' in agent.single_message:
                    agent.direction('w')
            self._log(f'unlock: {agent.message!r}')
            return
        if self.levitating():
            # no kicking while levitating (dokick.c: no leverage): come down first -- a potion's
            # levitation is waited out one square back, out of the eels' reach
            if self._timed_levitation():
                self._waiting = True
                self._set_state('waiting for the levitation to end')
                agent.search()
                return
            self._set_state('coming down to kick the door')
            self._stop_levitating()
            return
        if jf_config.BREACH_DOOR:
            # on Elbereth only what ignores it is worth a blow (hitting a scared monster erases it: mon.c setmangry)
            near = [m for m in self._hostiles_near(1) if self.dive._ignores_elbereth(m[3])]
            if near:
                _, y, x, mon, _ = near[0]
                self._set_state(f'fighting {getattr(mon, "mname", "?")} at the door')
                with agent.atom_operation():
                    agent.step(A.Command.FIGHT)
                    agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
                return
        elif self._fight_adjacent():
            return
        if agent.blstats.time < getattr(agent, '_no_kick_until', -1):
            agent.search(3)
            return
        tries['kick'] = tries.get('kick', 0) + 1
        if tries['kick'] > 60:
            self._give_up('the back door would not open')
            agent.search()
            return
        self._set_state('kicking the door')
        agent.kick(*to_bot(*DOOR))

    def _timed_levitation(self):
        """Levitating from a potion (it times out) rather than a ring or boots we can take off."""
        src = self._lev_source
        if src is not None and src[0] != 'potion':
            return False
        if jf_config.CASTLE_EDGE_REST:
            # only a KNOWN levitation ring/boots is a lasting lift: an unknown cursed ring left on after its test
            # 'may be' levitation and made a potion's lift look lasting -- pwc-dp9 jf14-s0 rested its potion away
            return not any(i.equipped and i.is_unambiguous() and i.object.name in ('levitation', 'levitation boots')
                           for i in self._items() if i.category == nh.RING_CLASS or i.is_armor())
        return not any('levitation' in self._names(i) and i.equipped for i in self._items()
                       if i.category == nh.RING_CLASS or i.is_armor())
