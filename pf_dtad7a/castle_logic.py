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

    def active(self):
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
        return self.levitating() or self.water_walking()

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
            if item.glyphs[0] in self._tested:
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
        self._tested.add(item.glyphs[0])
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
        agent.inventory.items.update(force=True)

    def _stop_levitating(self):
        """Come down (over the trap door this drops us through it: trap.c float_down -> dotrap)."""
        agent = self.agent
        tries = self._tries
        src = self._lev_source
        rings = [r for r in self._worn_rings() if 'levitation' in self._names(r)]
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
        nexts.sort(key=lambda p: (self._monster_at(*p), prefer_route and p not in ROUTE_INDEX))
        self._step_to(*nexts[0])
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
        return pos not in WEST_COURTYARD or self.levitating() or self.water_walking()

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
            spot = TEST_SPOT
            target = to_bot(*spot)
        if agent.bfs()[target] == -1:
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
            return self._approach(GOAL)
        if self._floating():
            # lifted already (a ring put on where we landed) but still in the west maze: to the corner
            return self._approach(CORNER)
        plan = self._plan()
        cold = self._cold_source()
        if not plan and cold is None:
            self._give_up('nothing that crosses water')
            return False
        pos = self._pos()
        self._guard_moat_edge()
        if pos in WEST_COURTYARD and self._wield_weapon():
            return True
        # Everything is tried one square inside the courtyard (a levitating hero in the maze throws daggers
        # and hurtles backwards into walls -- castle-c4 seed 1 'killed by bumping into a wall'): lasting
        # lifts first (known lev ring/boots, a wish), then untested rings/boots/amulets, then potions.
        # A cold ray can't carry us: each frozen moat square costs the ray 3 extra range (zap.c
        # zap_over_floor), so a zap freezes 2-3 squares of the 26 on the way round -- it only helps a
        # stranded crossing.
        kind, item = plan[0] if plan else ('cold', cold)
        spot = CORNER if kind in ('horn', 'cold') else TEST_SPOT
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
        if spot == TEST_SPOT and engraving != 'elbereth' and agent.can_engrave() and \
                not agent.character.prop.blind and not self._tries.get('spot_elbereth'):
            self._tries['spot_elbereth'] = 1
            agent.engrave('Elbereth')
            return True
        lasting = kind in ('wish', 'boots', 'ring', 'amulet')
        bl = agent.blstats
        if lasting and bl.hitpoints < 0.85 * bl.max_hitpoints and not self._hostiles_near(1) and \
                self._tries.get('rest', 0) < 300:
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
        want = 0.9 if (self._resting or self._after_valley_retreat()) else 0.45
        if bl.hitpoints >= want * bl.max_hitpoints or self._tries.get('rest_stop', 0) > 600:
            if self._resting:
                self._log(f'rested: hp {bl.hitpoints}/{bl.max_hitpoints}')
            self._resting = False
            return False
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
        if pos == self._last_pos:
            self._stuck += 1
        else:
            self._stuck = 0
        self._last_pos = pos
        self._log(f'x {pos} hp {agent.blstats.hitpoints} lev {int(self.levitating())} t {agent.blstats.time}')
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
        if self._rest_stop(pos):
            return
        if pos in (GOAL, DOOR, TRAPDOOR):
            self._door_step(pos)
            return
        if self._floating():
            if self._wield_weapon():
                return
            bl = agent.blstats
            if pos in WEST_COURTYARD and not self._timed_levitation() and \
                    bl.hitpoints < 0.85 * bl.max_hitpoints and self._tries.get('rest', 0) < 300:
                # floating on a ring/boots in the courtyard: set off at full strength (fight what comes)
                self._tries['rest'] = self._tries.get('rest', 0) + 1
                if not self._fight_adjacent():
                    self._set_state('resting before the crossing')
                    agent.search(3)
                return
            # one fixed way round (the north half), pressing on through whatever blocks it: sharks bite 5d6
            # and follow along the ring, so turning back only doubles the time in their reach
            dist = _bfs(GOAL, NORTH)
            if pos not in dist:
                dist = _bfs(GOAL, OUTSIDE)
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
            self._set_state(f'stranded at {pos}')
            self._try(*plan[0])
            return
        self._give_up(f'stranded at {pos}')
        agent.search()

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
        if self._fight_adjacent():
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
        return not any('levitation' in self._names(i) and i.equipped for i in self._items()
                       if i.category == nh.RING_CLASS or i.is_armor())
