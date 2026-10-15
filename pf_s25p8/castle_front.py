"""FRONT_DOOR (castle-front lane): the castle's front door as a banked gamble for a kit with no lift.

The castle depth is already scored when we stand on its level, so a death here costs nothing. A kit whose passage
plan gave up ('nothing that crosses water') and that holds a known wand of striking or opening goes in by the front:

* do.c boulder_hits_pool(): a boulder pushed into the moat fills it 90% of the time ('Now you can cross it!'); pushed
  into the raised span (05,08) it sets DB_FLOOR under the bridge. A cold ray over the span sets DB_ICE (zap.c).
* zap.c bhit(): a striking bolt east along row 08 passes the raised span and destroys the drawbridge at the
  portcullis wall (06,08). dbridge.c destroy_drawbridge(): the span becomes MOAT, or ROOM/ICE when filled or frozen
  first ('The drawbridge disintegrates!'); the portcullis a doorless doorway; iron-chain debris scatters within one
  square; wake_nearto(500) wakes everything within ~22 squares (the antechamber's 8 soldiers and lieutenant). A
  wand of opening at the portcullis lowers the bridge instead: a dry span at once.
  So: fill first, then open from (03,08), two squares off the span (debris, the drawbridge eels at (05,07)/(05,09)).
* The span's only castle-side neighbour is the doorway (06,08) ((06,07)/(06,09) are wall), so the soldiers (@: they
  ignore Elbereth) come out one at a time; we hold (04,08), where the one on the span is the only one in reach (and a
  polearm from the doorway).
* Then the antechamber (07..14,05..11), the closed door (15,08), the unlit hallway (16..25,08), the closed door
  (26,08) and the throne room (27..37,05..11) with 27 awake court monsters (L N E H M O R T X Z, all respect
  Elbereth). The castle's wand of wishing lies in a chest in one of the four corner towers (castle.des $place).

Milestones are logged as 'FRONT M:<name>' for the harness measurement (dev/scenarios/castle-front.json).
"""

import collections

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from .castle_logic import to_bot, to_map, map_char
from .glyph import G, SS
from .level import Level
from .strategy import Strategy

SPAN = (5, 8)          # the raised drawbridge (castle.des DRAWBRIDGE:(05,08),east), moat under it
PORTCULLIS = (6, 8)    # its wall: a doorless doorway once the bridge is destroyed or lowered
ZAP_SPOT = (3, 8)      # two squares short of the span: off the debris and out of the eels' reach
# where the soldiers are met as they come over the span: v1 held (04,08), the span's west neighbour, and the drawbridge
# eels (plus a crocodile) beside it drowned or mauled 3 of 3 harness holds within ~20 turns (fr1). (03,08) has no
# water beside it.
HOLD = (3, 8)
THRONE_ENTRY = (27, 8)
THRONE_ROOM = frozenset((x, y) for x in range(27, 38) for y in range(5, 12))
ANTE = frozenset((x, y) for x in range(7, 15) for y in range(5, 12))
PUSH_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR_NAME = {(1, 0): 'e', (-1, 0): 'w', (0, 1): 's', (0, -1): 'n'}
WEST_BOX = (-8, 4, 0, 16)     # map x0, x1, y0, y1 of the boulder search: the west maze and courtyard
SOLDIER_NAMES = ('soldier', 'sergeant', 'lieutenant', 'captain')
DRY_SYMS = frozenset({SS.S_room, SS.S_darkroom, SS.S_ice, SS.S_vodbridge, SS.S_hodbridge, SS.S_ndoor})
HOLD_QUIET = 15        # turns with no soldier in view before we go in
REST_HP = 0.5          # rest on Elbereth inside when below this share of HP (court monsters respect it)

# ---- FRONT_V3 (front-strong lane)
# The west maze's only link to the courtyard is (-1,10): castle.des MAZEWALK:(00,10),west starts sp_lev.c walkfrom()
# at map (0,10) = level (9,13) (odd, odd), which can only carve west: (-1,10) and the maze cell (-2,10). Maze cells
# sit on odd level coordinates, so (-2,10)'s diagonal neighbours (even, even) are always wall: standing there, the
# monster on (-1,10) is the only castle-side one that can reach us, and no moat square is next to us (the eels).
MOUTH = (-2, 10)
MOUTH_GATE = (-1, 10)
# The court's way out of the throne room is the door (26,08) into the one-wide hallway (16..25,08) (walls on rows 07
# and 09). Monsters head for us greedily (monmove.c m_move; set_apparxy: an awake one that can see knows where we are),
# so with us off row 08 most of the court sticks to the throne room's west wall: it is met on the way in. At (25,08) the
# one in the doorway is the only one that can reach us from the throne room ((26,07)/(26,09) are wall).
HALL_HOLD = (25, 8)
HOLD_MIN = 60          # hold steps at the mouth at least (the soldiers need ~5-40 turns to come out)
HOLD_REST = 0.5        # rest on Elbereth at the mouth below this share of HP while only Elbereth-respecting monsters are near
HOLD_LEAVE = 0.85      # the hold ends when nothing hostile has been in view for HOLD_QUIET turns and HP is back to this
HOLD_MAX = 700         # hold steps at most; then in, if HP >= HOLD_LEAVE
RETREAT_HP = 0.35      # west of the hallway (x <= 14), this badly hurt with a hostile next to us: back to the mouth
MAX_RETREATS = 6
MAX_LURES = 8          # times at most the way in turns back to the mouth because 2+ hostiles came into view
RESUME_HP = 0.75       # a rest on Elbereth lasts until this share of HP
TAME_HP = 0.5          # read a known scroll of taming when 2+ hostiles are next to us below this share of HP
THRONE_N = (32, 5)     # throne-room square under the locked door (32,04) to the north hallway (row 03)
THRONE_S = (32, 11)    # ... over the locked door (32,12) to the south hallway (row 13)
# castle.des: the wand of wishing lies in a chest in one of the four corner towers ($place: (04,02) (58,02) (04,14)
# (58,14)), on a burned Elbereth under a cursed scroll of scare monster; the other towers are empty. Tower: (hallway
# row, door, first square inside, chest square, interior)
TOWERS = {
    'NW': (3, (7, 3), (6, 3), (4, 2), frozenset((x, y) for x in range(2, 7) for y in (2, 3))),
    'NE': (3, (55, 3), (56, 3), (58, 2), frozenset((x, y) for x in range(56, 61) for y in (2, 3))),
    'SW': (13, (7, 13), (6, 13), (4, 14), frozenset((x, y) for x in range(2, 7) for y in (13, 14))),
    'SE': (13, (55, 13), (56, 13), (58, 14), frozenset((x, y) for x in range(56, 61) for y in (13, 14))),
}
TOWER_ORDER = ('NW', 'NE', 'SE', 'SW')


class FrontDoor:
    def __init__(self, dive):
        self.dive = dive
        self.agent = dive.agent
        self.done = False
        self.opened = False
        self.tries = collections.Counter()
        self.quiet = 0
        self.logged = set()
        self.dead_boulders = set()   # boulders a push could not move (something under or behind them for good)
        self._state = None
        # FRONT_V3
        self.resting = False
        self.hold_over = False
        self.hold_i = 0              # 0: the maze mouth, 1: the hallway square (25,08)
        self.hold_steps = 0
        self.retreats = 0
        self.lures = 0
        self.scare_at = set()        # map squares where we dropped a scroll of scare monster
        self.phase = None            # 'wand' once in the throne room
        self.tower_i = 0             # index into TOWER_ORDER
        self.chest_fail = 0

    # ------------------------------------------------------------------ helpers

    def _log(self, msg):
        self.agent.log(f'FRONT {msg}')

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self._log(f'{state} (pos {self._pos()}, turn {self.agent.blstats.time})')

    def _mile(self, name, extra=''):
        """A measurement milestone, logged once."""
        if name in self.logged:
            return
        self.logged.add(name)
        bl = self.agent.blstats
        self._log(f'M:{name} turn {bl.time} hp {bl.hitpoints}/{bl.max_hitpoints} pos {self._pos()} {extra}'.rstrip())

    def _stop(self, why):
        if not self.done:
            self.done = True
            self._mile('stop', why)

    def _pos(self):
        bl = self.agent.blstats
        return to_map(bl.y, bl.x)

    def _sym(self, p):
        y, x = to_bot(*p)
        level = self.agent.current_level()
        if not (0 <= y < level.objects.shape[0] and 0 <= x < level.objects.shape[1]):
            return None
        return level.objects[y, x]

    def _glyph(self, p):
        y, x = to_bot(*p)
        return self.agent.glyphs[y, x]

    def _boulder(self, p):
        y, x = to_bot(*p)
        level = self.agent.current_level()
        if not (0 <= y < level.objects.shape[0] and 0 <= x < level.objects.shape[1]):
            return False
        return self.agent.glyphs[y, x] in G.BOULDER or level.objects[y, x] in G.BOULDER

    def _walkable(self, p):
        if jf_config.FRONT_V3 and p in (SPAN, PORTCULLIS) and self._open() and self._span_dry():
            return True   # the bot's map may still hold the moat/wall there (fs3c-k3 s0: '(6, 8) unreachable')
        y, x = to_bot(*p)
        level = self.agent.current_level()
        if not (0 <= y < level.walkable.shape[0] and 0 <= x < level.walkable.shape[1]):
            return False
        return bool(level.walkable[y, x])

    def _monster_at(self, p):
        return self.dive.castle._monster_at(*p)

    def on_castle(self):
        c = self.dive.castle
        level = self.agent.current_level()
        return c.castle_key is not None and level.key() == c.castle_key and \
            level.dungeon_number == Level.DUNGEONS_OF_DOOM

    def _usable(self, name):
        c = self.dive.castle
        return next((i for i in c._items() if c._usable_wand(i, name)), None)

    def _opener(self):
        """A known wand of opening (lowers the bridge: no fill needed) or striking (destroys it)."""
        for name in ('opening', 'striking'):
            it = self._usable(name)
            if it is not None and self.tries[('zap', name)] < 4:
                return it
        return None

    def _open(self):
        if self.opened:
            return True
        return self._sym(PORTCULLIS) == SS.S_ndoor or self._sym(SPAN) in (SS.S_vodbridge, SS.S_hodbridge)

    def _span_dry(self):
        if jf_config.FRONT_V3 and getattr(self, 'span_filled', False):
            return True   # 'Now you can cross it!' from a push onto the span (a monster standing on it hides the floor)
        return self._sym(SPAN) in DRY_SYMS or self._glyph(SPAN) in DRY_SYMS

    def _hostiles_adjacent(self):
        bl = self.agent.blstats
        return [m for m in self.agent.get_visible_monsters()
                if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) == 1 and
                not (jf_config.FRONT_V3 and nh.glyph_is_pet(int(m[4])))]
        # (FRONT_V3: a monster our scroll of taming tamed is no target -- fs7-k6 s8 killed its tame minotaur,
        # -15 alignment, and Excalibur's blast killed us)

    def _soldiers_in_view(self):
        """@ in the antechamber, the doorway, on the span or on our side of the moat."""
        out = []
        for m in self.agent.get_visible_monsters():
            name = getattr(m[3], 'mname', '')
            p = to_map(m[1], m[2])
            if name in SOLDIER_NAMES and (p in ANTE or p in (PORTCULLIS, SPAN) or p[0] <= 4):
                out.append(m)
        return out

    # ------------------------------------------------------------------ eligibility

    def active(self):
        """Cheap test: on the castle's west side with the lift plan given up (or none), a way to open the front."""
        if not jf_config.FRONT_DOOR or self.done or not self.on_castle():
            return False
        if self.dive.castle.active():
            return False   # the passage plan (a lift, cold, a wish) goes first
        agent = self.agent
        if agent.character.prop.polymorph:
            return False
        if self._pos()[0] >= 57 and self.phase != 'wand':
            return False   # east side: the back door is the castle logic's (the NE/SE towers are x 56-60)
        return self._open() or self._opener() is not None

    def strategy(self):
        """Keeps control for the whole front-door plan; the emergency layer (prayer, potions) and the minotaur guard
        still preempt it."""
        def f():
            if not self.active():
                yield False
            yield True
            steps = 0
            while self.active() and steps < 300:
                before = self.agent.step_count
                if not self._step():
                    break
                steps += 1
                if self.agent.step_count == before:
                    self.agent.search()

        return Strategy(f)

    # ------------------------------------------------------------------ the plan

    def _step(self):
        if jf_config.FRONT_V3:
            return self._step_v3()
        agent = self.agent
        pos = self._pos()
        if not self.logged:
            inv = '; '.join(i.text for i in self.dive.castle._items())
            self._mile('start', f'open={self._open()} dry={self._span_dry()} inv: {inv}')
        if self._span_dry():
            self._mile('filled', f'span {self._sym(SPAN)}')
        if self._open() and self._span_dry():
            self._mile('path', 'span dry, portcullis open')
            if pos in THRONE_ROOM or pos[0] >= THRONE_ENTRY[0]:
                self._mile('throne')
                self._stop('reached the throne room (v1 stops here)')
                return False
            if agent.wield_best_melee_weapon():
                return True   # the pick-axe may be in hand after a dig
            if 'inside' not in self.logged and self._holding():
                return True
            return self._enter_step()
        if not self._span_dry():
            if self._fill_step():
                return True
            if self.done:
                return False
        return self._open_step()

    # ---- fill the span (before opening: the soldiers stay asleep until the path is ready)

    def _fill_step(self):
        opener = self._opener()
        if not self._open() and opener is not None and opener.object.name == 'opening':
            return False   # a lowered bridge needs no fill
        plan = self._plan_push()
        if plan:
            return self._push_step(plan)
        cold = self._usable('cold')
        if cold is not None and self.tries['cold'] < 3:
            if self._pos() != ZAP_SPOT:
                return self._go(ZAP_SPOT, 'to the zap square to freeze the span')
            self.tries['cold'] += 1
            self._set_state('freezing the span with a cold ray')
            self.agent.zap(cold, 'e')
            self._log(f'cold zap: {self.agent.message[:160]!r}')
            return True
        self._stop('no way to fill the span (no boulder push, no cold)')
        return False

    def _plan_push(self):
        """[(boulder square, push direction), ...] that roll a known boulder onto the span, or None. A Sokoban BFS
        over (boulder square, the hero's region) on the known dry floor of the west maze and courtyard: orthogonal
        pushes only (a diagonal push between walls fails when carrying > 600), other boulders are walls, and no
        water but the span receives the boulder."""
        x0, x1, y0, y1 = WEST_BOX
        squares = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
        boulders = {p for p in squares if self._boulder(p)}
        if not boulders - self.dead_boulders:
            return None
        floor = {p for p in squares if (self._walkable(p) or p in boulders) and map_char(*p) != '}'}
        me = self._pos()
        floor.add(me)
        best = None
        for b0 in sorted(boulders - self.dead_boulders, key=lambda p: abs(p[0] - SPAN[0]) + abs(p[1] - SPAN[1])):
            walls = boulders - {b0}
            path = self._push_bfs(b0, me, floor - walls)
            if path is not None and (best is None or len(path) < len(best)):
                best = path
        return best

    @staticmethod
    def _reach(start, free):
        seen = {start}
        todo = [start]
        while todo:
            x, y = todo.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (x + dx, y + dy)
                if n in free and n not in seen:
                    seen.add(n)
                    todo.append(n)
        return seen

    def _push_bfs(self, b0, h0, floor):
        start_reach = self._reach(h0, floor - {b0})
        key0 = (b0, min(start_reach))
        prev = {key0: None}
        todo = collections.deque([(b0, h0, key0)])
        while todo:
            b, h, key = todo.popleft()
            reach = self._reach(h, floor - {b})
            for d in PUSH_DIRS:
                q = (b[0] - d[0], b[1] - d[1])
                t = (b[0] + d[0], b[1] + d[1])
                if q not in reach:
                    continue
                if t == SPAN:
                    out = [(b, d)]
                    k = key
                    while prev[k] is not None:
                        k, push = prev[k]
                        out.append(push)
                    return out[::-1]
                if t not in floor or map_char(*t) == '}':
                    continue
                nreach = self._reach(b, floor - {t})
                nkey = (t, min(nreach))
                if nkey in prev:
                    continue
                prev[nkey] = (key, (b, d))
                todo.append((t, b, nkey))
        return None

    def _push_step(self, plan):
        agent = self.agent
        b, d = plan[0]
        q = (b[0] - d[0], b[1] - d[1])
        self._mile('fill_plan', f'{len(plan)} pushes, first {b} {DIR_NAME[d]}')
        if self._pos() != q:
            return self._go(q, f'to push the boulder at {b} {DIR_NAME[d]}')
        key = ('push', b, d)
        self.tries[key] += 1
        if self.tries[key] > 8:
            self._log(f'push {b} {DIR_NAME[d]} keeps failing')
            return self._give_boulder_up(b)
        t = (b[0] + d[0], b[1] + d[1])
        if self._monster_at(t) and t != SPAN:
            self._set_state(f'waiting: a monster behind the boulder at {t}')
            return self._fight_or_wait()
        self._set_state(f'pushing the boulder at {b} {DIR_NAME[d]}')
        agent.direction(DIR_NAME[d])
        msg = agent.message
        if 'Now you can cross it' in msg:
            self._mile('filled', 'boulder')
            if (b[0] + d[0], b[1] + d[1]) == SPAN:
                self.span_filled = True
        elif 'sinks without a trace' in msg:
            self._log('the boulder sank without filling the span')
        elif 'in vain' in msg or 'cannot move' in msg:
            self._log(f'push failed: {msg[:120]!r}')
            if 'monster behind' in msg or "There's" in msg:
                agent.search()   # an unseen swimmer or walker behind it: give it a turn to move on
        return True

    def _give_boulder_up(self, b):
        """A boulder we can't move (a monster behind it for good, something under it): plan with the others."""
        self.dead_boulders.add(b)
        self._log(f'boulder at {b} will not move: planning without it')
        return True

    # ---- open the front

    def _open_step(self):
        agent = self.agent
        opener = self._opener()
        if opener is None:
            self._stop('no opener left')
            return False
        if self._pos() != ZAP_SPOT:
            return self._go(ZAP_SPOT, 'to the zap square')
        name = opener.object.name
        if jf_config.FRONT_V3 and self.tries['pet_in_line'] < 20:
            # a pet in the bolt's way (fs8-k6 s8: the minotaur our scroll of taming had just tamed): wait -- killing
            # it costs 15 alignment, and with the record below 0 Excalibur blasts its wielder
            for x in range(ZAP_SPOT[0] + 1, PORTCULLIS[0] + 1):
                y, bx = to_bot(x, ZAP_SPOT[1])
                if nh.glyph_is_pet(int(self.agent.glyphs[y, bx])):
                    self.tries['pet_in_line'] += 1
                    self._set_state(f'waiting: a pet in the line of the zap at {(x, ZAP_SPOT[1])}')
                    agent.search()
                    return True
        self.tries[('zap', name)] += 1
        self._set_state(f'zapping {name} at the drawbridge')
        agent.zap(opener, 'e')
        msg = agent.message
        self._log(f'{name} zap: {msg[:200]!r}')
        if any(s in msg for s in ('drawbridge disintegrates', 'falls into the moat', 'drawbridge collapses',
                                  'loud *CRASH*', 'loud *SPLASH*', 'drawbridge coming down', 'drawbridge going down')):
            self.opened = True
            self._mile('open', name)
        return True

    # ---- hold the doorway, then in

    def _holding(self):
        """After the bridge opened: hold HOLD until no soldier has been in view for HOLD_QUIET turns. On Elbereth
        while only monsters that respect it are next to us (the throne room's xorns walk through the walls; every court
        monster, eel and xorn respects it); the soldiers and minotaurs don't, so they are fought (which wipes it, 3.6:
        written again once none is adjacent)."""
        agent = self.agent
        near = self._hostiles_adjacent()
        ignorers = [m for m in near if self.dive._melee_ignores_elbereth(m[3])]
        soldiers = self._soldiers_in_view()
        if near or soldiers:
            self.quiet = 0
        else:
            self.quiet += 1
        if self.quiet >= HOLD_QUIET:
            self._mile('inside', 'hold over: no soldier in view')
            return False
        if self._pos() != HOLD:
            if self._monster_at(HOLD):
                return self._fight_or_wait()
            return self._go(HOLD, 'to the hold square')
        if ignorers:
            return self._attack(self._pick_target(ignorers))
        engraving = (agent.inventory.engraving_below_me or '').lower()
        if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind and \
                self.tries['hold_elbereth'] < 40:
            self.tries['hold_elbereth'] += 1
            self._set_state('holding: Elbereth')
            agent.engrave('Elbereth')
            return True
        if near and engraving != 'elbereth':
            return self._attack(self._pick_target(near))
        self._set_state('holding' + (' on Elbereth' if engraving == 'elbereth' else ''))
        agent.search()
        return True

    def _pick_target(self, near):
        span = to_bot(*SPAN)
        for m in near:
            if (m[1], m[2]) == span:
                return m
        for m in near:
            if getattr(m[3], 'mname', '') in SOLDIER_NAMES:
                return m
        return near[0]

    def _attack(self, m):
        agent = self.agent
        _, y, x, mon, _ = m
        if jf_config.FRONT_V3 and self._engraved() and not self.dive._melee_ignores_elbereth(mon):
            # mon.c setmangry(): a blow at a monster that respects Elbereth, struck from an Elbereth square, costs 5
            # alignment ('You feel like a hypocrite.' -- prayers fail below 0) and deletes the engraving anyway:
            # write over it first
            self.tries['unwrite'] += 1
            self._set_state(f'wiping our Elbereth before fighting {getattr(mon, "mname", "?")}')
            agent.engrave('x')
            return True
        self._set_state(f'fighting {getattr(mon, "mname", "?")} at {to_map(y, x)}')
        with agent.atom_operation():
            agent.step(A.Command.FIGHT)
            agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
        return True

    def _fight_or_wait(self):
        near = self._hostiles_adjacent()
        if near:
            return self._attack(self._pick_target(near))
        self.agent.search()
        return True

    def _enter_step(self):
        """Along row 08 into the throne room: through the doorway, the antechamber, the doors (15,08) and (26,08)
        (closed, not locked: moving into them opens them) and the hallway; fight what stands in the way, rest on
        Elbereth when hurt and nothing that ignores it is near."""
        agent = self.agent
        bl = agent.blstats
        pos = self._pos()
        self._mile('inside')
        near = self._hostiles_adjacent()
        if bl.hitpoints < REST_HP * bl.max_hitpoints and not any(
                self.dive._ignores_elbereth(m[3]) for m in agent.get_visible_monsters()) and \
                self.tries['rest'] < 400:
            self.tries['rest'] += 1
            engraving = (agent.inventory.engraving_below_me or '').lower()
            if engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind:
                self._set_state('Elbereth to rest inside')
                agent.engrave('Elbereth')
                return True
            self._set_state('resting inside')
            agent.search(3)
            return True
        if near:
            return self._attack(self._pick_target(near))
        if pos[1] != 8 or pos[0] < PORTCULLIS[0]:
            target = (pos[0] + 1, 8) if pos[1] == 8 else (max(pos[0], PORTCULLIS[0]), 8)
            return self._go(target, 'back to row 08')
        nxt = (pos[0] + 1, 8)
        if self._monster_at(nxt):
            return self._fight_or_wait()
        self._set_state(f'walking east along row 08 at {pos}')
        before = agent.blstats.time
        agent.direction('e')
        msg = agent.message
        if 'This door is locked' in msg:
            self.tries[('kick', nxt)] += 1
            if self.tries[('kick', nxt)] > 12:
                self._stop(f'locked door at {nxt}')
                return False
            with agent.atom_operation():
                agent.step(A.Command.KICK)
                agent.direction('e')
        elif self._pos() == pos and agent.blstats.time == before:
            self.tries[('stuck', pos)] += 1
            if self.tries[('stuck', pos)] > 10:
                self._stop(f'stuck at {pos}: {msg[:80]!r}')
                return False
        return True

    def _door(self, p):
        return self._sym(p) in G.DOORS

    def _path(self, start, target):
        """Shortest walk over squares the level map knows as walkable (not boulders), one step at a time: the
        agent's BFS keeps off the moat-edge squares castle_logic forbids, and those are exactly where we work.
        No diagonal step through a door or squeezed between two walls (> 600 weight: 'carrying too much')."""
        prev = {start: None}
        todo = collections.deque([start])
        while todo:
            p = todo.popleft()
            if p == target:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (p[0] + dx, p[1] + dy)
                if n in prev or not (-8 <= n[0] <= 70 and -3 <= n[1] <= 17):
                    continue
                if n != target and (not self._walkable(n) or self._boulder(n)):
                    continue
                if n == target and self._boulder(n):
                    continue
                if dx and dy and (self._door(p) or self._door(n) or
                                  (not self._walkable((p[0] + dx, p[1])) and not self._walkable((p[0], p[1] + dy)))):
                    continue
                prev[n] = p
                todo.append(n)
        if target not in prev:
            return None
        path = []
        p = target
        while p != start:
            path.append(p)
            p = prev[p]
        return path[::-1]

    def _go(self, p, why):
        agent = self.agent
        here = self._pos()
        self._set_state(f'walking {why} {p}')
        path = self._path(here, p)
        if not path:
            self.tries[('unreachable', p)] += 1
            if jf_config.FRONT_V3 and self.tries[('unreachable', p)] in (1, 10):
                rows = []
                for yy in range(max(-3, min(here[1], p[1]) - 2), min(17, max(here[1], p[1]) + 3)):
                    row = ''
                    for xx in range(max(-8, min(here[0], p[0]) - 2), min(70, max(here[0], p[0]) + 3)):
                        q = (xx, yy)
                        row += '@' if q == here else '*' if q == p else 'B' if self._boulder(q) else \
                            'M' if self._monster_at(q) else 'w' if self._walkable(q) else '#'
                    rows.append(row)
                self._log(f'no path {here} -> {p}: ' + ' | '.join(rows))
            if self.tries[('unreachable', p)] > 20:
                self._stop(f'{p} unreachable')
                return False
            if jf_config.FRONT_V3 and self._push_toward(p):
                return True
            if p[0] <= 4:
                return self.dive.castle._approach(p)   # explores (or digs) the west maze to the courtyard
            return self._fight_or_wait()
        n = path[0]
        if self._monster_at(n):
            if jf_config.FRONT_V3:
                return self._attack_at(n)   # the blocker itself (v2 hit whatever stood on the span)
            return self._fight_or_wait()
        y, x = to_bot(*n)
        agent.direction(agent.calc_direction(agent.blstats.y, agent.blstats.x, y, x))
        return True

    # ------------------------------------------------------------------ FRONT_V3 (front-strong lane)

    def _step_v3(self):
        agent = self.agent
        pos = self._pos()
        if not self.logged:
            inv = '; '.join(i.text for i in self.dive.castle._items())
            self._mile('start', f'open={self._open()} dry={self._span_dry()} inv: {inv}')
        if self._span_dry():
            self._mile('filled', f'span {self._sym(SPAN)}')
            self.span_filled = True   # a monster on the span later hides the floor (fs4-k4 s7 stopped on it)
        if self._held() and self._held_escape():
            return True
        if not (self._open() and self._span_dry()):
            # before the front is open: an attacker next to us (the west maze's minotaur, an early @) is fought
            # first -- v2 pushed boulders under a minotaur's blows (fsb-k3 s3) -- from a dropped scroll of scare
            # monster when we hold one and it ignores Elbereth; moat monsters are left alone
            near = [m for m in self._hostiles_adjacent() if map_char(*to_map(m[1], m[2])) != '}']
            bl = agent.blstats
            minos = [m for m in self._land_hostiles() if getattr(m[3], 'mname', '') == 'minotaur' and
                     max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 3]
            if self._on_scare() and minos and self.tries['scare_hold_pre'] < 300:
                # on the pile with the minotaur about: strike it from here (scared, it can't melee us), don't walk off
                self.tries['scare_hold_pre'] += 1
                if near:
                    return self._attack(self._pick_target(near))
                agent.search()
                return True
            if near:
                # a scroll for a minotaur only, once: the holds need two (fs4-k3 dropped all three on the way to
                # the boulders, one per @ that came next to us)
                if not self._on_scare() and self.tries['scare_pre'] < 1 and \
                        any(getattr(m[3], 'mname', '') == 'minotaur' for m in near) and self._drop_scare():
                    self.tries['scare_pre'] += 1
                    return True
                if self._read_taming(near):
                    return True
                return self._attack(self._pick_target(near))
            if not self._span_dry():
                if self._fill_step():
                    return True
                if self.done:
                    return False
            return self._open_step()
        self._mile('path', 'span dry, portcullis open')
        if agent.wield_best_melee_weapon():
            return True   # the pick-axe may be in hand after a dig
        if self.phase == 'wand':
            return self._wand_step()
        if not self.hold_over:
            return self._hold_v3()
        if self.hold_i == 0:
            return self._advance(HALL_HOLD, 1)
        if pos in THRONE_ROOM:
            self.phase = 'wand'
            self._mile('throne')
            return self._wand_step()
        return self._advance(THRONE_ENTRY, None)

    def _held(self):
        msg = self.agent.message or ''
        if 'swings itself around you' in msg or 'cannot escape from' in msg:
            self._held_turn = self.agent.blstats.time
        held_turn = getattr(self, '_held_turn', None)
        return held_turn is not None and self.agent.blstats.time - held_turn <= 3

    def _held_escape(self):
        """Held by a sea monster (the drawbridge eels reach the span, the portcullis and the maze gate; fs4-k3 s1/s5
        drowned there): Elbereth, as CFP_EEL -- a scared holder lets go (monmove.c distfleeck -> monflee ->
        release_hero) and doesn't attack, while its next touch from the water drowns us (mhitu.c AD_WRAP)."""
        agent = self.agent
        if self._on_scare() or agent.character.prop.blind or not agent.can_engrave():
            return False
        if self._engraved():
            if 'You get released' in (agent.message or '') or self.tries['held_wait'] >= 8:
                return False
            self.tries['held_wait'] += 1
            self._set_state('held on Elbereth: waiting for the holder to let go')
            agent.search()
            return True
        if self.tries['held_elbereth'] >= 12:
            return False
        self.tries['held_elbereth'] += 1
        self._set_state('held by a sea monster: Elbereth')
        agent.engrave('Elbereth')
        self._log(f'held at {self._pos()}: Elbereth -> {agent.message[:80]!r}')
        return True

    def _push_toward(self, p):
        """No path: a boulder next to us on the way (fsdbg-k1 s1: one on the maze gate (-1,10), a giant's throw) is
        pushed on, orthogonally first, when the square beyond it is floor."""
        here = tuple(int(v) for v in self._pos())
        sx = int(p[0] > here[0]) - int(p[0] < here[0])   # (numpy bools don't subtract)
        sy = int(p[1] > here[1]) - int(p[1] < here[1])
        for d in [(sx, 0), (0, sy), (sx, sy)]:
            if d == (0, 0):
                continue
            n = (here[0] + d[0], here[1] + d[1])
            beyond = (n[0] + d[0], n[1] + d[1])
            if not self._boulder(n) or self._boulder(beyond) or not self._walkable(beyond):
                continue
            y, x = to_bot(*n)
            dname = self.agent.calc_direction(self.agent.blstats.y, self.agent.blstats.x, y, x)
            if self.tries[('pushon', n, d)] >= 2 or self._monster_at(beyond):
                # 'You hear a monster behind the boulder' (fs6t-k3 s0/s18: a giant's boulder on the maze gate, a
                # pudding behind it): a force bolt fractures it (zap.c bhito: fracture_rock)
                wand = self._usable('striking')
                if wand is not None and self.tries[('boulder_zap', n)] < 2:
                    self.tries[('boulder_zap', n)] += 1
                    self._set_state(f'zapping striking {dname} at the boulder {n} in the way to {p}')
                    self.agent.zap(wand, dname)
                    self._log(f'boulder zap: {self.agent.message[:120]!r}')
                    return True
                if self.tries[('pushon', n, d)] >= 6:
                    continue
            self.tries[('pushon', n, d)] += 1
            self._set_state(f'pushing the boulder at {n} out of the way to {p}')
            self.agent.direction(dname)
            return True
        return False

    def _near(self):
        """Hostiles next to us, not counting the moat's swimmers (fighting an eel from the bank is a losing trade)."""
        out = []
        for m in self._hostiles_adjacent():
            p = to_map(m[1], m[2])
            if map_char(*p) == '}' and p != SPAN:
                continue
            out.append(m)
        return out

    def _attack_at(self, p):
        """The monster on our path at p: a hostile is fought; a peaceful one is waited out (then fought)."""
        y, x = to_bot(*p)
        if nh.glyph_is_pet(int(self.agent.glyphs[y, x])):
            # our pet: walking into it swaps places
            self.agent.direction(self.agent.calc_direction(self.agent.blstats.y, self.agent.blstats.x, y, x))
            return True
        for m in self.agent.get_visible_monsters():
            if (m[1], m[2]) == (y, x):
                return self._attack(m)
        self.tries[('blocked', p)] += 1
        if self.tries[('blocked', p)] > 30:
            self._log(f'{p} blocked for 30 steps: fighting it')
            self._set_state(f'fighting the blocker at {p}')
            with self.agent.atom_operation():
                self.agent.step(A.Command.FIGHT)
                self.agent.direction(self.agent.calc_direction(self.agent.blstats.y, self.agent.blstats.x, y, x))
            return True
        near = self._near()
        if near:
            return self._attack(self._pick_target(near))
        self.agent.search()
        return True

    def _advance(self, target, next_hold):
        """On to target (the hallway hold, the throne room): fights on the way; hurt with only Elbereth-respecting
        monsters in view -> Elbereth and rest; badly hurt with an attacker next to us -> back to the hold behind us."""
        pos = self._pos()
        hp = self._hp_frac()
        near = self._near()
        if near and self._read_taming(near):
            return True
        if near and hp < RETREAT_HP and self.retreats < MAX_RETREATS:
            self._retreat(near)
            return self._hold_v3()
        if next_hold == 1 and pos[0] <= 14 and self.lures < MAX_LURES:
            # the court comes out once we are on row 08 (greedy m_move toward us): 2+ hostiles in view on the way
            # in -> back to the mouth and take them there one at a time
            bl = self.agent.blstats
            crowd = [m for m in self._land_hostiles() if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 8]
            if len(crowd) >= 2:
                self.lures += 1
                self.hold_over = False
                self.hold_steps = 0
                self.quiet = 0
                self._log(f'lure {self.lures} at {pos}: {[getattr(m[3], "mname", "?") for m in crowd]} in view, back '
                          f'to the mouth')
                return self._hold_v3()
        if self._rest_inside():
            return True
        if pos == target and next_hold is not None:
            self.hold_i = next_hold
            self.hold_over = False
            self.hold_steps = 0
            self.quiet = 0
            return self._hold_v3()
        if near:
            return self._attack(self._pick_target(near))
        if pos[1] == 8 and PORTCULLIS[0] <= pos[0] < target[0] and target[1] == 8:
            # along row 08: the closed doors (15,08) and (26,08) open as we walk into them (a BFS doesn't cross them)
            return self._walk_dir('e', f'along row 08 to {target}')
        if pos[1] == 8 and pos[0] == target[0] + 1 and target[1] == 8:
            return self._walk_dir('w', f'back to {target}')
        if pos[0] <= 14:
            return self._go((max(pos[0], PORTCULLIS[0]), 8), f'back to row 08 on the way to {target}')
        return self._go(target, f'on to {target}')

    def _retreat(self, near):
        self.retreats += 1
        self.hold_over = False
        self.hold_steps = 0
        self.quiet = 0
        self._log(f'retreat {self.retreats} to hold {self.hold_i} from {self._pos()}, hp {self._hp_frac():.2f}, next to '
                  f'us: {[getattr(m[3], "mname", "?") for m in near]}')

    def _hp_frac(self):
        bl = self.agent.blstats
        return bl.hitpoints / max(1, bl.max_hitpoints)

    def _engraved(self):
        return (self.agent.inventory.engraving_below_me or '').lower() == 'elbereth'

    def _can_write(self):
        return self.agent.can_engrave() and not self.agent.character.prop.blind

    def _land_hostiles(self):
        """Visible hostiles not in the moat (its eels and sharks never come ashore)."""
        out = []
        for m in self.agent.get_visible_monsters():
            p = to_map(m[1], m[2])
            if (map_char(*p) == '}' and p != SPAN) or nh.glyph_is_pet(int(m[4])):
                continue
            out.append(m)
        return out

    def _known_scroll(self, name):
        for it in self.agent.inventory.items:
            if it.category == nh.SCROLL_CLASS and it.is_unambiguous() and it.object.name == name:
                return it
        return None

    def _on_scare(self):
        return self._pos() in self.scare_at

    def _drop_scare(self):
        """A known scroll of scare monster dropped on our square: nothing melees us there, @ and minotaurs included
        (monmove.c onscary: no exceptions for the scroll), and our blows don't wipe it. It stays behind (a second
        pickup would turn it to dust)."""
        from . import power
        known, _ = power.scare_scrolls(self.agent)
        if not known or self.tries['scare_drop'] >= 4:
            return False
        self.tries['scare_drop'] += 1
        pos = self._pos()
        self._set_state(f'dropping a scroll of scare monster at {pos}')
        self.scare_at.add(pos)   # before the drop: its atomic step's preempt checks may switch strategy (castle B005)
        self.agent.inventory.drop(known[:1], [1])
        self.agent.inventory._note_dropped(known[:1], [1], force=True)
        self._log(f'scare scroll dropped at {pos}: {self.agent.message[:100]!r}')
        self._mile('scare', f'at {pos}')
        return True

    def _read_taming(self, near):
        """Astra's emergency: a known scroll of taming read with 2+ hostiles next to us while hurt (soldiers have no
        magic resistance to resist it; the court's xorns and trolls fight for us after)."""
        if len(near) < 2 or self._hp_frac() >= TAME_HP or self.tries['taming'] >= 3:
            return False
        it = self._known_scroll('taming')
        if it is None:
            return False
        self.tries['taming'] += 1
        letter = self.agent.inventory.items.get_letter(it)
        self._set_state(f'reading taming, next to us: {[getattr(m[3], "mname", "?") for m in near]}')
        with self.agent.atom_operation():
            self.agent.step(A.Command.READ)
            self.agent.type_text(letter)
        self._log(f'taming: {self.agent.message[:160]!r}')
        self._mile('taming')
        return True

    def _hold_square(self):
        if self.hold_i == 1:
            return HALL_HOLD
        if self.tries[('unreachable', MOUTH)] > 20 or self._boulder(MOUTH):
            return HOLD
        return MOUTH

    def _hold_v3(self):
        """Hold the west maze's mouth until the castle has stopped coming: the monster on (-1,10) is the only one
        that can reach us from the castle side. Elbereth-ignorers (@, minotaurs) are fought; the court (all respect
        Elbereth) is fought while HP >= HOLD_REST, below that we rest on Elbereth (written again whenever a blow of
        ours has scuffed it and nothing is next to us)."""
        agent = self.agent
        hold = self._hold_square()
        pos = self._pos()
        hp = self._hp_frac()
        near = self._near()
        self.hold_steps += 1
        self.quiet = 0 if self._land_hostiles() else self.quiet + 1
        min_steps = HOLD_MIN if self.hold_i == 0 else 0
        if hp >= HOLD_LEAVE and ((self.quiet >= 2 * HOLD_QUIET and self.hold_steps >= min_steps) or
                                 self.hold_steps >= HOLD_MAX):
            self.hold_over = True
            self._mile('inside' if self.hold_i == 0 else 'hall_done',
                       f'hold {self.hold_i} over after {self.hold_steps} steps (quiet {self.quiet})')
            self._log(f'hold {self.hold_i} over: quiet {self.quiet}, steps {self.hold_steps}, hp {hp:.2f}')
            return True
        if pos != hold:
            return self._go(hold, f'to hold {self.hold_i}')
        self._mile('hold' if self.hold_i == 0 else 'hall_hold', f'at {hold}')
        on_scare = self._on_scare()
        if not on_scare and self._drop_scare():
            return True
        self._update_resting(HOLD_REST)
        if near:
            if self._read_taming(near):
                return True
            if on_scare:
                return self._attack(self._pick_target(near))
            ignorers = [m for m in near if self.dive._melee_ignores_elbereth(m[3])]
            if ignorers:
                return self._attack(self._pick_target(ignorers))
            if self.resting:
                if self._engraved():
                    self._set_state('resting on Elbereth at the hold')
                    agent.search()
                    return True
                if self._can_write() and self.tries['hold_elbereth'] < 400:
                    self.tries['hold_elbereth'] += 1
                    self._set_state('Elbereth at the hold')
                    agent.engrave('Elbereth')
                    return True
            return self._attack(self._pick_target(near))   # (an Elbereth left under us is wiped first)
        if self.resting and not on_scare and not self._engraved() and self._can_write() and \
                self.tries['hold_elbereth'] < 400:
            self.tries['hold_elbereth'] += 1
            self._set_state('Elbereth at the hold')
            agent.engrave('Elbereth')
            return True
        self._set_state('holding' + (' on the scroll' if on_scare else ' on Elbereth' if self._engraved() else ''))
        agent.search()
        return True

    # ---- the wand: throne room -> hallway -> corner tower -> chest

    def _wand_in_pack(self):
        """The wand we took from the tower chest (any wand we didn't carry before the chest)."""
        before = getattr(self, '_wands_before', None)
        if before is None:
            return None
        for it in self.agent.inventory.items:
            if it.is_wand() and self.agent.inventory.items.get_letter(it) not in before:
                return it
        return None

    def _update_resting(self, below):
        """Rest from below `below` of max HP until RESUME_HP (a blow at an Elbereth-respecting monster from the
        Elbereth square costs alignment: fights resume only once healed, the engraving wiped first)."""
        hp = self._hp_frac()
        if hp < below:
            self.resting = True
        elif hp >= RESUME_HP:
            self.resting = False

    def _rest_inside(self):
        """Hurt and nothing that ignores Elbereth in view: Elbereth, then rest (as _enter_step)."""
        agent = self.agent
        self._update_resting(REST_HP)
        if not self.resting or self.tries['rest'] >= 600 or \
                any(self.dive._ignores_elbereth(m[3]) for m in agent.get_visible_monsters()):
            return False
        self.tries['rest'] += 1
        if not self._engraved() and self._can_write():
            self._set_state('Elbereth to rest')
            agent.engrave('Elbereth')
            return True
        self._set_state('resting')
        agent.search(3)
        return True

    def _wand_step(self):
        pos = self._pos()
        wand = self._wand_in_pack()
        if wand is not None:
            return self._test_wand(wand)
        near = self._near()
        if near and self._read_taming(near):
            return True
        if near and self._hp_frac() < RETREAT_HP and pos in THRONE_ROOM and self.retreats < MAX_RETREATS:
            self.phase = None
            self.hold_i = 1
            self._retreat(near)
            return self._hold_v3()
        if self._rest_inside():
            return True
        if near and pos not in [TOWERS[t][3] for t in TOWER_ORDER]:
            return self._attack(self._pick_target(near))
        if self.tower_i >= len(TOWER_ORDER):
            self._stop('no tower chest found')
            return False
        name = TOWER_ORDER[self.tower_i]
        row, door, inside, chest, room = TOWERS[name]
        if pos in room or pos == door:
            return self._tower_step(name)
        for oname, (orow, odoor, oinside, ochest, oroom) in TOWERS.items():
            # still in (or in the door of) a tower we are done with: out onto its hallway
            if pos in oroom:
                if pos != oinside:
                    return self._go(oinside, f'back to the {oname} tower door')
                return self._walk_dir('e' if odoor[0] > oinside[0] else 'w', f'out of the {oname} tower')
            if pos == odoor:
                return self._walk_dir('e' if odoor[0] < 32 else 'w', f'out of the {oname} door onto the hallway')
        if pos[1] == row and 8 <= pos[0] <= 54:
            self._mile('hallway', f'row {row}')
            return self._walk_dir('w' if door[0] < pos[0] else 'e', f'along the hallway to the {name} door {door}')
        north = row == 3
        tdoor = (32, 4) if north else (32, 12)
        if pos == tdoor:
            return self._walk_dir('n' if north else 's', 'out of the throne-room door')
        if pos[1] in (3, 13) and 8 <= pos[0] <= 54:
            # the other hallway: back to its throne-room door
            if pos[0] != 32:
                return self._walk_dir('w' if pos[0] > 32 else 'e', 'back along the hallway')
            return self._walk_dir('s' if pos[1] == 3 else 'n', 'back into the throne room')
        under = THRONE_N if north else THRONE_S
        if pos != under:
            return self._go(under, 'to the throne-room door')
        return self._walk_dir('n' if north else 's', f'through the throne-room door {tdoor}')

    def _zap_door(self, d):
        wand = self._usable('striking')
        if wand is None or self.tries[('door_zap', d)] >= 3:
            return False
        self.tries[('door_zap', d)] += 1
        self._set_state(f'zapping striking {d} at a locked door')
        self.agent.zap(wand, d)
        self._log(f'door zap {d}: {self.agent.message[:120]!r}')
        return True

    def _walk_dir(self, d, why):
        agent = self.agent
        pos = self._pos()
        dx, dy = {'n': (0, -1), 's': (0, 1), 'e': (1, 0), 'w': (-1, 0)}[d]
        nxt = (pos[0] + dx, pos[1] + dy)
        if self._monster_at(nxt):
            return self._attack_at(nxt)
        self._set_state(f'walking {d} {why}')
        before = agent.blstats.time
        agent.direction(d)
        msg = agent.message
        if 'This door is locked' in msg:
            self.tries[('kick', nxt)] += 1
            if self._zap_door(d):
                return True
            if self.tries[('kick', nxt)] > 30:
                self._stop(f'locked door at {nxt}')
                return False
            with agent.atom_operation():
                agent.step(A.Command.KICK)
                agent.direction(d)
            self._log(f'kick {nxt}: {agent.message[:100]!r}')
        elif self._pos() == pos and agent.blstats.time == before:
            self.tries[('stuck', pos, d)] += 1
            if self.tries[('stuck', pos, d)] > 10:
                self._stop(f'stuck at {pos} going {d}: {msg[:80]!r}')
                return False
        return True

    def _objects_at(self, p):
        y, x = to_bot(*p)
        g = self.agent.glyphs[y, x]
        return bool(nh.glyph_is_object(g)) or self.agent.current_level().item_count[y, x] > 0

    def _tower_step(self, name):
        row, door, inside, chest, room = TOWERS[name]
        pos = self._pos()
        if pos == door:
            return self._walk_dir('w' if inside[0] < door[0] else 'e', f'into the {name} tower')
        self._mile('tower', name)
        if pos == chest:
            return self._chest_step()
        if self._objects_at(chest):
            self._mile('chest_seen', f'{name} {chest}')
            return self._go(chest, f'to the {name} tower chest')
        if self._glyph(chest) in (SS.S_room, SS.S_darkroom):
            self._log(f'the {name} tower is empty')
            self.tower_i += 1
            return True
        self.tries[('look', name)] += 1
        if self.tries[('look', name)] > 15:
            self._log(f'the {name} chest square stays unseen: next tower')
            self.tower_i += 1
            return True
        near = self._near()
        if near:
            return self._attack(self._pick_target(near))
        return self._go(chest, f'toward the {name} chest square')

    def _chest_step(self):
        """On the chest (a burned Elbereth and the cursed scroll of scare monster under us: a refuge): #force the
        lock with the wielded blade (lock.c: oc_wldam * 2 % per turn, an artifact resists breaking), #loot, take
        the wand out."""
        from .exceptions import AgentPanic
        agent = self.agent
        inv = agent.inventory
        self._mile('chest')
        if self.chest_fail > 12:
            self._stop('could not get the wand out of the chest')
            return False
        if getattr(self, '_wands_before', None) is None:
            self._wands_before = {inv.items.get_letter(i) for i in inv.items if i.is_wand()}
        try:
            below = inv.get_items_below_me() or inv.items_below_me or []
        except AgentPanic as e:
            self.chest_fail += 1
            self._log(f'look failed: {e}')
            return True
        chest = next((i for i in below if i.is_container() or i.is_possible_container()), None)
        if chest is not None and (agent.hands_welded() or 'free hand' in (agent.message or '')):
            # a cursed (welded) weapon beside a shield: no #loot, no #force (fs5-k3 s7: a court caster's curse items;
            # 150 'Without a free hand' asserts). The prayer fixes TROUBLE_UNUSEABLE_HANDS.
            self.chest_fail += 1
            if agent.is_safe_to_pray():
                self._log('hands welded on the chest: praying')
                agent.pray()
            else:
                agent.search()
            return True
        if chest is None:
            self.chest_fail += 1
            self._log(f'no chest below us: {[i.text for i in below]}')
            if below or self.chest_fail >= 3:
                # someone's loot on the corner square (fs3c-k3 s1): not the wand's tower
                self.tower_i += 1
                self.chest_fail = 0
                return True
            agent.search()
            return True
        try:
            inv.check_container_content(chest)
        except (AgentPanic, AssertionError) as e:
            self.chest_fail += 1
            self._log(f'chest check failed: {e}')
            return True
        content = chest.content
        if content is not None and content.locked:
            self.tries['force'] += 1
            if self.tries['force'] > 25:
                self.chest_fail += 100
                return True
            self._set_state('forcing the chest lock')
            with agent.atom_operation():
                agent.step(A.Command.FORCE)
                if 'force its lock?' in agent.message or 'force its lid?' in agent.message or '[ynq]' in agent.message:
                    agent.type_text('y')
            self._log(f'force: {agent.message[:160]!r}')
            return True
        items = list(content.items) if content is not None else []
        wands = [i for i in items if i.is_wand()]
        self._log(f'chest content: {[i.text for i in items]}')
        if not wands:
            self.chest_fail += 1
            return True
        try:
            inv.use_container(chest, [], wands[:1])
        except (AgentPanic, AssertionError) as e:
            self.chest_fail += 1
            self._log(f'take-out failed: {e}')
            return True
        inv.items.update(force=True)
        if self._wand_in_pack() is not None:
            self._mile('wand', self._wand_in_pack().text)
        return True

    def _test_wand(self, wand):
        """The tower chest's wand is the wand of wishing: engrave with it (zapnodir: the wish; WISH_TELEPORT_ROUTE
        takes over from there)."""
        from . import castle_power
        self._mile('wand', wand.text)
        if wand.is_unambiguous() or self.tries['wand_test'] >= 2:
            self._stop(f'wand in hand: {wand.text!r}')
            return False
        self.tries['wand_test'] += 1
        self._set_state(f'engrave-testing the chest wand {wand.text!r}')
        castle_power._engrave_test(self.dive.castle, wand)
        self._mile('wish', f'{self.agent.message[:120]!r}')
        return True


def strategy(dive):
    front = getattr(dive, 'front', None)
    if front is None:
        return Strategy(lambda: iter([False]))
    return front.strategy()
