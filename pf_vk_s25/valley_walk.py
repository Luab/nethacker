"""valley-walk lane: the Valley of the Dead on foot for a castle-crossing kit (jf_config.VALLEY_WALK, off by default).

A castle crossing lands us in the Valley (Gehennom 1, depth castle+1). With the castle at Dlvl 25-28 that depth is
0.507-0.647; the score passes 0.647 only one level below the Valley. The Valley is hardfloor and noteleport, so the
way on is its '>' in the far north-west corner (valley.py): 105-167 steps through Moloch's temple and three locked
secret doors, a pick-axe breaks each in 3 dig steps.

Mechanics the walk uses (NetHack 3.6.6 source):
- monmove.c set_apparxy(): a monster that can see knows our square wherever it is; m_move() steps greedily toward it
  (dist2, no path finding) or along our trail (gettrack) -- so every resident heads for us, and walls hold some.
- monmove.c dochug(): a monster that moved in its action does not melee in that action (m_move() == 1 returns before
  mattacku). Walking away from a pursuer no faster than us takes almost no blows; standing to fight it costs one attack
  round per monster per action.
- Valkyrie intrinsic Fast (XL 7): speed 12 plus a free action one turn in three (~16). Vampire bats 20 (can't be
  outrun), vampire lords 14, giant mummies 14, other mummies 10-12, zombies 6-8, ghouls 6, liches 6-9, ghosts 3.
- M2_STALK (monst.c): mummies, bats (a vampire in bat form too), liches and ghouls never follow us up the '<';
  vampires in V form, zombies, wraiths and ghosts do when adjacent and not fleeing (do.c keepdogs).
- monmove.c disturb(): a Valkyrie's intrinsic Stealth never wakes the graveyards' sleepers (mkroom.c fill_zoo,
  MM_ASLEEP on every one of the 122 squares); only attacking one does.
- hack.c monster_nearby(): any awake hostile next to us stops the dig occupation after each dig step (allmain.c runs
  the step first), so a door needs ~3 quiet dig steps (a dwarf's effort doubles each step, dig.c).

The plan (one high-priority strategy that owns every Valley move while it holds; the old valley_step/valley_sneak/
fight2 take over when it yields, e.g. hallucinating or with no route known):
1. the '>' in reach: go there, down at once (the dig-dive below banks each next level);
2. at a door's dig square: clear what stands next to us, then dig;
3. otherwise one step along the cheapest way to the next objective: unseen graveyard squares, sleepers, awake
   monsters and the squares next to awake monsters cost extra, so the way bends round them when it can;
4. fight only what blocks that step, what can't be outrun (a bat next to us), or everything when boxed in;
5. VALLEY_WALK_HOLD: on landing first stand on the '<' (1-13 steps from every landing square) and fight the landing
   crowd there with the climb at hand (the castle mode heals us upstairs and drops us back through the trap door).
"""

import heapq

import nle.nethack as nh
import numpy as np

from . import jf_config, utils, valley
from . import objects as O
from .glyph import MON
from .strategy import Strategy

MM_SCROLL = O.from_name('magic mapping', nh.SCROLL_CLASS)
ID_SCROLL = O.from_name('identify', nh.SCROLL_CLASS)

# monsters we never melee on the way (passive paralysis/acid/stoning, explosions); the walk goes round them
NO_MELEE = frozenset(('floating eye', 'gelatinous cube', 'acid blob', 'yellow mold', 'green mold', 'brown mold',
                      'red mold', 'lichen', 'spotted jelly', 'blue jelly', 'ochre jelly', 'cockatrice',
                      'chickatrice', 'gas spore'))
# makemon difficulty-ish danger: these take most of an XL-8 hero's HP in a stand-up fight (vr-base killers)
DANGEROUS = frozenset(('giant mummy', 'ettin mummy', 'human mummy', 'vampire lord', 'vampire', 'ghoul', 'lich',
                       'demilich', 'master lich', 'arch-lich', 'giant zombie', 'ettin zombie', 'xorn', 'minotaur',
                       'troll', 'ice troll', 'rock troll', 'Olog-hai', 'wraith', 'barbed devil', 'bone devil',
                       'horned devil', 'ice devil', 'vrock', 'hezrou', 'nalfeshnee', 'marilith', 'pit fiend',
                       'balrog', 'jabberwock', 'purple worm', 'mind flayer', 'master mind flayer', 'unknown'))


def _name(mon):
    return getattr(mon, 'mname', '') or ''


def _speed(mon):
    return int(getattr(mon, 'mmove', 12) or 0)


class ValleyWalker:
    """Per-game state of the Valley walk; the dive owns one (dive.walker)."""

    def __init__(self, dive):
        self.dive = dive
        self.agent = dive.agent
        self._visit = None           # (level key, first turn) of the current Valley visit
        self._phase = None           # 'hold' | 'walk'
        self._hold_start = None
        self._hostile_turn = -1      # last turn a hostile was in view
        self._seen_at = {}           # (y, x) -> (glyph, first turn seen there)
        self._attackers = {}         # monster name -> last turn it hit/missed us
        self._grave_clear = set()    # graveyard squares seen empty (their sleeper is dead or never was)
        self._last_target = None
        self._logged = None
        self._idle = 0
        self._cooldown_until = -1
        self._climbs = 0
        self._last_attack = None     # (y, x) of our last melee target
        self._lottery_reads = 0      # VALLEY_LOTTERY
        self._lottery_quaffs = 0
        self._lottery_cooldown = -1
        self._lottery_idle = 0

    # ------------------------------------------------------------------ bookkeeping (every step, no bfs)

    def active(self):
        return jf_config.VALLEY_WALK and self.dive.in_valley() and not self.dive._valley_misplaced

    def update(self):
        """Dive update hook (every step in the Valley): who stands where since when, graveyard squares seen
        empty, who attacked us."""
        agent = self.agent
        if not self.active():
            return
        bl = agent.blstats
        turn = bl.time
        key = agent.current_level().key()
        if self._visit is None or self._visit[0] != key or \
                (self.dive._valley_arrival is not None and self._visit[1] < self.dive._valley_arrival):
            self._visit = (key, turn)
        mask = agent.monster_tracker.monster_mask
        glyphs = agent.glyphs
        seen = {}
        for y, x in zip(*mask.nonzero()):
            p = (int(y), int(x))
            g = int(glyphs[y, x])
            old = self._seen_at.get(p)
            seen[p] = old if old is not None and old[0] == g else (g, turn)
        self._seen_at = seen
        y0, x0 = int(bl.y), int(bl.x)
        for y in range(y0 - 1, y0 + 2):
            for x in range(x0 - 1, x0 + 2):
                if (y, x) in valley.GRAVEYARD:
                    if mask[y, x]:
                        self._grave_clear.discard((y, x))
                    else:
                        self._grave_clear.add((y, x))
        for m in self.dive._ATTACK_MSG.finditer(agent.message or ''):
            self._attackers[m.group(1) or m.group(2)] = turn
        if turn // 10 != getattr(self, '_status_turn', -1):
            # diagnostics: where the walk is, every 10 turns
            self._status_turn = turn // 10
            near = sorted((max(abs(int(y) - y0), abs(int(x) - x0)), int(y), int(x))
                          for y, x in zip(*mask.nonzero()) if (int(y), int(x)) != (y0, x0))[:6]
            names = []
            for d, y, x in near:
                try:
                    names.append(f'{MON.permonst(int(glyphs[y, x])).mname}@{d}')
                except Exception:
                    names.append(f'?@{d}')
            self._log(f'status t{turn} phase={self._phase} pos={(y0, x0)} hp {bl.hitpoints}/{bl.max_hitpoints} '
                      f'xl {bl.experience_level} near={names}')

    def _sleeper(self, m):
        """A monster on a graveyard square that has stood there since we first saw it and hasn't attacked us lately:
        one of the fill's sleepers (they never move until attacked)."""
        _, y, x, mon, glyph = m
        p = (int(y), int(x))
        if p not in valley.GRAVEYARD:
            return False
        last = self._attackers.get(_name(mon))
        if last is not None and self.agent.blstats.time - last <= 5:
            return False
        return True

    # ------------------------------------------------------------------ planning helpers

    def _log(self, msg):
        self.agent.log(f'VWALK {msg}')

    def invisible(self):
        """Invisible and not seeing it: display.c shows no hero glyph on our square (canspotself() false)."""
        agent = self.agent
        g = int(agent.glyphs[agent.blstats.y, agent.blstats.x])
        return not (nh.glyph_is_monster(g) or nh.glyph_is_pet(g))

    def _our_speed(self):
        return 16   # intrinsic Fast; Very_fast (speed boots/potion) is ~20 -- not tracked yet

    def _danger(self, mon):
        name = _name(mon)
        if name in DANGEROUS or getattr(mon, 'mlevel', 0) >= 9:
            return 3
        return 1

    # the level file's fixed traps on the only way (valley.py): the bot's BFS leaves a known trap out (and never
    # enters a sleeping gas trap: agent.UNSAFE_TRAPS), but the pits guard 1-wide corridors and the gas trap is the
    # square before the '>' -- the walk goes through them
    FIXED_TRAPS = frozenset(valley.SPIKED_PITS + (valley.SLEEP_GAS,))

    def _reach(self, dis, pos):
        level = self.agent.current_level()
        reach = dis >= 0
        reach[pos] = True
        for t in self.FIXED_TRAPS:
            if level.walkable[t]:
                reach[t] = True
        return reach

    def _objective(self, level, dis, pos, reach, cost):
        """('down', [square], dist) when the '>' is reachable, ('door', door, stands, dist) for the next door to
        open, else None (dist: _dijkstra toward the objective)."""
        down = valley.DOWN_STAIRS
        if pos == down:
            return ('down', [down], None)
        dist = self._dijkstra([down], reach, cost)
        if np.isfinite(dist[pos]):
            return ('down', [down], dist)
        nxt = self.dive._valley_next_door(level)
        if nxt is None:
            return None
        door, stands = nxt
        usable = [s for s in stands if reach[s]]
        if pos in stands:
            return ('door', door, list(stands), None)
        if not usable:
            return None
        dist = self._dijkstra(usable, reach, cost)
        if not np.isfinite(dist[pos]):
            return None
        return ('door', door, usable, dist)

    def _cost_map(self, monsters, pos):
        agent = self.agent
        level = agent.current_level()
        shape = level.walkable.shape
        cost = np.ones(shape)
        for sq in valley.GRAVEYARD:
            if sq not in self._grave_clear:
                cost[sq] += jf_config.VALLEY_WALK_GRAVE_COST
        peaceful = agent.monster_tracker.peaceful_monster_mask
        cost[peaceful] += 10
        for m in monsters:
            y, x = int(m[1]), int(m[2])
            if self._sleeper(m):
                cost[y, x] += jf_config.VALLEY_WALK_SLEEPER_COST
                continue
            d = self._danger(m[3])
            cost[y, x] += 6 * d
            # the squares next to an awake monster: stepping there gives it a blow without its having to move
            if _speed(m[3]) > 0:
                y1, y2 = max(y - 1, 0), min(y + 2, shape[0])
                x1, x2 = max(x - 1, 0), min(x + 2, shape[1])
                cost[y1:y2, x1:x2] += jf_config.VALLEY_WALK_ADJ_COST * d
        for trap in valley.SPIKED_PITS + (valley.SLEEP_GAS,):
            cost[trap] += 2
        cost[pos] = 1
        return cost

    def _dijkstra(self, targets, reach, cost):
        """dist[s] = cheapest cost of the squares after s on the way to any target (cost of entering each square)."""
        h, w = reach.shape
        dist = np.full((h, w), np.inf)
        heap = []
        for t in targets:
            dist[t] = 0.0
            heap.append((0.0, t))
        heapq.heapify(heap)
        squeeze = self.agent.inventory.items.total_weight <= 600
        while heap:
            d, (y, x) = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            nd = d + cost[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < h and 0 <= nx < w and reach[ny, nx] and nd < dist[ny, nx] and \
                            (not (dy and dx) or squeeze or reach[ny, x] or reach[y, nx]):
                        dist[ny, nx] = nd
                        heapq.heappush(heap, (nd, (ny, nx)))
        return dist

    def _best_step(self, pos, dis, reach, cost, dist):
        """(square, value) of the cheapest first step along dist (a _dijkstra map); None when no step leads on."""
        best, best_v = None, np.inf
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (pos[0] + dy, pos[1] + dx)
                if (not (dy or dx)) or not (0 <= n[0] < reach.shape[0] and 0 <= n[1] < reach.shape[1]):
                    continue
                # a legal single step for the bot's own BFS (walls, doorway diagonals, squeezes), or one of the fixed
                # traps it leaves out
                if dis[n] != 1 and not (n in self.FIXED_TRAPS and reach[n]):
                    continue
                v = cost[n] + dist[n]
                if v < best_v:
                    best, best_v = n, v
        return (best, best_v) if best is not None and np.isfinite(best_v) else None

    def _step_toward(self, targets, pos, dis, reach, cost):
        dist = self._dijkstra(targets, reach, cost)
        return self._best_step(pos, dis, reach, cost, dist)

    # ------------------------------------------------------------------ actions

    def _attack(self, m, why):
        agent = self.agent
        if agent.wield_best_melee_weapon():
            return
        y, x = int(m[1]), int(m[2])
        self._last_attack = (y, x)
        self._log(f'attack {_name(m[3])} at {(y, x)} ({why}) hp {agent.blstats.hitpoints}/{agent.blstats.max_hitpoints}')
        agent.melee_attack(y, x)

    def _descend(self):
        agent = self.agent
        dive = self.dive
        if dive.levitating():
            dive._valley_descend(valley.DOWN_STAIRS)   # takes the levitation ring off / waits
            return
        bl = agent.blstats
        self._log(f'down the stairs at turn {bl.time} hp {bl.hitpoints}/{bl.max_hitpoints} '
                  f'({bl.time - (dive._valley_arrival or bl.time)} turns in the Valley)')
        agent.move('>')

    def _climb(self, why):
        agent = self.agent
        bl = agent.blstats
        self._climbs += 1
        self.dive.valley_retreats += 1
        self.dive.valley_retreat_turn = bl.time
        self._phase = None
        self._log(f'climb {self._climbs} up the stairs ({why}) hp {bl.hitpoints}/{bl.max_hitpoints} turn {bl.time}')
        agent.move('<')

    # ------------------------------------------------------------------ the decision

    def _decide(self):
        """One action as (kind, args) or None (the old Valley logic acts)."""
        agent = self.agent
        dive = self.dive
        bl = agent.blstats
        prop = agent.character.prop
        if prop.hallu or prop.blind or prop.stun or prop.confusion or prop.polymorph:
            return None
        level = agent.current_level()
        pos = (int(bl.y), int(bl.x))
        hp, maxhp = bl.hitpoints, bl.max_hitpoints
        monsters = agent.get_visible_monsters()
        if monsters:
            self._hostile_turn = bl.time
        adjacent = [m for m in monsters if utils.adjacent((int(m[1]), int(m[2])), pos)]
        awake_adj = [m for m in adjacent if not self._sleeper(m)]
        dis = agent.bfs()
        up = valley.UP_STAIRS
        reach = self._reach(dis, pos)
        cost = self._cost_map(monsters, pos)
        obj = self._objective(level, dis, pos, reach, cost)

        # the '>' first: whatever is around, the level below is banked progress
        if obj is not None and obj[0] == 'down' and pos == obj[1][0]:
            return ('descend',)

        # the landing hold on the '<'
        if jf_config.VALLEY_WALK_HOLD and self._phase in (None, 'hold') and (pos == up or 0 <= dis[up] <= 15):
            if self._phase is None:
                self._phase = 'hold'
                self._hold_start = bl.time
            quiet = bl.time - max(self._hostile_turn, self._hold_start)
            held = bl.time - self._hold_start
            if (quiet >= jf_config.VALLEY_WALK_HOLD_QUIET and hp >= jf_config.VALLEY_WALK_HOLD_LEAVE * maxhp) or \
                    held >= jf_config.VALLEY_WALK_HOLD_MAX:
                self._phase = 'walk'
                self._log(f'hold over after {held} turns (quiet {quiet}) hp {hp}/{maxhp}: walking')
            else:
                if pos != up:
                    blocker = next((m for m in adjacent if not self._sleeper(m)), None)
                    step = self._step_toward([up], pos, dis, reach, cost)
                    if step is None:
                        return ('attack', blocker, 'boxed on the way to the stairs') if blocker else None
                    n = step[0]
                    on_n = next((m for m in monsters if (int(m[1]), int(m[2])) == n), None)
                    if on_n is not None:
                        return ('attack', on_n, 'blocks the way to the stairs')
                    return ('move', n)
                # on the '<'
                bad = [m for m in awake_adj if self._danger(m[3]) >= 3]
                if hp < jf_config.VALLEY_WALK_CLIMB_BELOW * maxhp and awake_adj:
                    return ('climb', f'hp low with {[_name(m[3]) for m in awake_adj]} next to us')
                if len(bad) >= 2 or (bad and hp < 0.75 * maxhp):
                    return ('climb', f'{[_name(m[3]) for m in bad]} next to us')
                if awake_adj:
                    target = min(awake_adj, key=lambda m: (_speed(m[3]) < 15, self._danger(m[3])))
                    return ('attack', target, 'holding the stairs')
                if hp < jf_config.VALLEY_WALK_CLIMB_BELOW * maxhp and monsters:
                    return ('climb', 'hp low, hostiles coming')
                return ('wait',)

        if self._phase is None:
            self._phase = 'walk'

        # hurt near the '<': up to heal (the castle mode rests to 90% and drops us back in)
        if hp < jf_config.VALLEY_WALK_RETREAT_BELOW * maxhp and awake_adj and \
                (pos == up or 0 <= dis[up] <= jf_config.VALLEY_WALK_RETREAT_REACH):
            if pos == up:
                return ('climb', 'hurt on the walk')
            step = self._step_toward([up], pos, dis, reach, cost)
            if step is not None:
                n = step[0]
                on_n = next((m for m in monsters if (int(m[1]), int(m[2])) == n), None)
                return ('attack', on_n, 'blocks the retreat') if on_n is not None else ('move', n)

        # a monster faster than us next to us can't be outrun -- unless we are invisible: then a monster within 6
        # squares in our line of sight moves at random 10 times in 11 (monmove.c m_move: should_see && Invis &&
        # !perceives -> appr = 0; no Valley resident sees invisible), and each melee swing goes where it guesses we are
        # (set_apparxy disp 1: right ~40% of the time once we have moved) -- walking on beats standing to swat bats
        fast = [m for m in awake_adj if _speed(m[3]) > 15 and _name(m[3]) not in NO_MELEE]
        if fast and not (jf_config.VALLEY_WALK_INVIS_WALK and self.invisible()):
            target = next((m for m in fast if (int(m[1]), int(m[2])) == self._last_attack), fast[0])
            return ('attack', target, 'too fast to outrun')

        if obj is None:
            return None   # valley_step explores for the next door / the '>'

        if obj[0] == 'door' and pos in obj[2]:
            if awake_adj:
                target = min(awake_adj, key=lambda m: self._danger(m[3]))
                return ('attack', target, 'at the door')
            return ('dig', obj[1])

        step = self._best_step(pos, dis, reach, cost, obj[-1])
        if step is None:
            if awake_adj:
                return ('attack', awake_adj[0], 'boxed in')
            return None
        n = step[0]
        on_n = next((m for m in monsters if (int(m[1]), int(m[2])) == n), None)
        if on_n is not None:
            if _name(on_n[3]) in NO_MELEE:
                return ('wait',)
            return ('attack', on_n, 'sleeper in the way' if self._sleeper(on_n) else 'in the way')
        if agent.monster_tracker.monster_mask[n]:
            return ('wait',)   # a peaceful (the priest of Moloch) on the square: let him move
        return ('move', n)

    def _act(self, action):
        agent = self.agent
        kind = action[0]
        bl = agent.blstats
        if self._logged != (kind, bl.time):
            self._logged = (kind, bl.time)
        if kind == 'descend':
            self._descend()
        elif kind == 'attack':
            self._attack(action[1], action[2])
        elif kind == 'move':
            agent.move(*action[1])
        elif kind == 'climb':
            self._climb(action[1])
        elif kind == 'dig':
            self.dive._valley_open_door(action[1])
        elif kind == 'wait':
            agent.search(1)
        else:
            raise ValueError(kind)

    # ------------------------------------------------------------------ VALLEY_LOTTERY

    def lottery_active(self):
        if not jf_config.VALLEY_LOTTERY or not self.dive.in_valley():
            return False
        agent = self.agent
        # a strong character walks out often enough (XL 14 + GDSM + speed boots: 12/42, R084) that a roll which goes up
        # 3 times in 4 costs more than it brings; the castle-crossing kits are XL 7-10 (0/61 walk out)
        if agent.blstats.experience_level > jf_config.VALLEY_LOTTERY_MAX_XL:
            return False
        # a polymorph form: VALLEY_XORN walks a wall-walker out; other forms may have no hands to read with
        if agent.character.prop.polymorph:
            return False
        from . import power_route
        # teleport control known: TC_ROUTE's own jump (a controlled level teleport answers 60: Gehennom's bottom-1)
        return not (jf_config.TC_ROUTE and power_route.tc_known(agent))

    def _mm_scrolls(self):
        """Known scrolls of magic mapping: on the Valley (nommap) one confuses the reader for rnd(30) turns (read.c)."""
        return [i for i in self.agent.inventory.items if i.category == nh.SCROLL_CLASS and i.is_unambiguous() and
                i.object == MM_SCROLL]

    def _lottery_plan(self):
        """The next lottery action, or None. teleport.c random_teleport_level() from depth V: 1 in 5 nothing, else
        uniform over 1..V-1 and V+1..V+3 -- P(deeper) = 0.8 * 3 / (V + 2) per read (~8%), and a trip up costs nothing
        the score has banked. A scroll of teleportation level-teleports when read cursed or confused (read.c); read
        uncursed and unconfused on the noteleport Valley it only says 'A mysterious force prevents you from
        teleporting!' and is identified (so the rest of its stack becomes a known ticket)."""
        from . import power_route
        agent = self.agent
        prop = agent.character.prop
        if prop.stun or prop.hallu:
            return None   # a stunned level prompt / hallucinated menus: wait
        tele = power_route.tele_scrolls(agent)
        if tele and prop.confusion:
            return ('read', tele[0], 'confused: a level teleport (the Valley lottery)')
        cursed = power_route.cursed_tele_scrolls(agent)
        if cursed:
            return ('read', cursed[0], 'cursed: a level teleport (the Valley lottery)')
        if prop.confusion:
            # confused, every unknown scroll that is teleportation is a ticket too. The one confused read that kills
            # is genocide (read.c do_genocide: confused = our own role, ~3.5% of random scrolls); punishment, fire,
            # create monster, remove curse and the rest are harmless or mild confused -- and the kit's odds without a
            # ticket are ~0 (VALLEY_LOTTERY_CONFUSED_UNKNOWN)
            if jf_config.VALLEY_LOTTERY_CONFUSED_UNKNOWN and self._lottery_reads < jf_config.VALLEY_LOTTERY_MAX_READS:
                unknown = [i for i in power_route.unknown_scrolls(agent) if i.text and 'unlabeled' not in i.text]
                if unknown:
                    it = max(unknown, key=lambda i: (power_route._p(i, (power_route.TELE_SCROLL,)), i.count))
                    return ('read', it, 'confused: an unknown scroll may be teleportation (a ticket)')
            return None
        if tele:
            srcs = power_route._confusion_sources(agent)
            if srcs:
                kind, it = srcs[0]
                return (kind, it, 'get confused for the lottery')
            mm = self._mm_scrolls()
            if mm:
                return ('read', mm[0], 'magic mapping on a nommap level: confusion for the lottery')
        unknown = [i for i in power_route.unknown_scrolls(agent) if i.text and 'unlabeled' not in i.text]
        pots = [i for i in agent.inventory.items if i.category == nh.POTION_CLASS and not i.is_unambiguous()]
        if unknown and jf_config.VALLEY_LOTTERY_CONFUSE_FIRST and pots and \
                self._lottery_quaffs < jf_config.VALLEY_LOTTERY_MAX_QUAFFS:
            # confused first, so that every unknown teleport scroll read after it is a ticket (read unconfused it is
            # only identified and spent: 46 of 138 salted kits so): an unknown potion is confusion or booze ~8% each
            it = max(pots, key=lambda i: power_route._p(i, (power_route.CONFUSION_POTION, power_route.BOOZE_POTION)))
            return ('quaff', it, 'unknown potion: may be confusion or booze (before reading the unknown scrolls)')
        if unknown and self._lottery_reads < jf_config.VALLEY_LOTTERY_MAX_READS:
            # identify first (it may name the teleport stack without spending one), then the likeliest teleportation
            # or magic mapping (a confusion source)
            it = max(unknown, key=lambda i: (power_route._p(i, (ID_SCROLL, power_route.TELE_SCROLL, MM_SCROLL)),
                                             i.count))
            return ('read', it, 'unknown scroll: may be teleportation (a ticket) or magic mapping (confusion)')
        if tele and jf_config.VALLEY_LOTTERY_QUAFF and self._lottery_quaffs < jf_config.VALLEY_LOTTERY_MAX_QUAFFS:
            if pots:
                it = max(pots, key=lambda i: power_route._p(i, (power_route.CONFUSION_POTION,
                                                               power_route.BOOZE_POTION)))
                return ('quaff', it, 'unknown potion: may be confusion or booze (for the lottery)')
        return None

    def _lottery_ok_now(self):
        """Read at a quiet moment (no awake hostile next to us), or at any moment once the fight is going badly: the
        Valley's odds without the lottery are ~0 for this kit."""
        agent = self.agent
        bl = agent.blstats
        pos = (int(bl.y), int(bl.x))
        adj = [m for m in agent.get_visible_monsters() if utils.adjacent((int(m[1]), int(m[2])), pos) and
               not self._sleeper(m)]
        return not adj or bl.hitpoints < jf_config.VALLEY_LOTTERY_DESPERATE * bl.max_hitpoints

    def _lottery_act(self, action):
        agent = self.agent
        kind, it, why = action
        bl = agent.blstats
        before = (bl.dungeon_number, bl.level_number, bl.depth)
        if kind == 'read':
            self._lottery_reads += 1
            self._read(it, why)
        else:
            self._lottery_quaffs += 1
            self._log(f'LOTTERY quaff {it.text!r}: {why}')
            self._quaff(it)
            self._log(f'LOTTERY quaffed -> {(agent.message or "")[:160]!r}')
        after = agent.blstats
        now = (after.dungeon_number, after.level_number, after.depth)
        if now != before:
            self._log(f'LOTTERY level teleport {before} -> {now} ({"deeper" if now[2] > before[2] else "up"})')

    def _quaff(self, item):
        """Quaff an unknown potion, closing what it may open: monster/object detection's browse_map getpos ('(For
        instructions type a '?')' -- left open, the monster tracker's next look read the wrong popup: vw-lot3-dev
        jf14-s6 ValueError after a potion of monster detection) and --More-- prompts."""
        from nle.nethack import actions as A
        agent = self.agent
        letter = agent.inventory.items.get_letter(item)
        answered = set()

        def gen():
            if 'What do you want to drink?' not in (agent.single_message or ''):
                return
            yield letter
            for _ in range(30):
                obs = agent._observation
                misc = obs['misc']
                top = ' '.join(bytes(line).decode('latin-1').replace('\0', ' ') for line in obs['tty_chars'][:2])
                msg = (agent.single_message or '') + ' ' + top
                if 'For instructions type a' in msg and 'getpos' not in answered:
                    answered.add('getpos')
                    yield A.Command.ESC
                    continue
                if 'Drink from the fountain' in msg or 'Dip it into the' in msg:
                    yield 'n'
                    continue
                if misc[2] or '--More--' in top:
                    yield ' '
                    continue
                return

        with agent.atom_operation():
            agent.step(A.Command.QUAFF, gen())
        agent.inventory.items.update(force=True)

    def _read(self, item, why):
        """Read a scroll, answering its prompts: genocide ('minotaur'; blessed: the class 'H' -- the maze levels below
        the Valley), the identify menu (power_route's picks), a stinking cloud's aim and a charging scroll (declined)."""
        from . import power_route
        from nle.nethack import actions as A
        agent = self.agent
        letter = agent.inventory.items.get_letter(item)
        bl = agent.blstats
        self._log(f'LOTTERY reading {item.text!r} ({letter}): {why}, hp {bl.hitpoints}/{bl.max_hitpoints}')
        menu_pages = {}
        answered = set()

        def gen():
            if 'What do you want to read?' not in agent.single_message:
                return
            yield letter
            for _ in range(60):
                obs = agent._observation
                misc = obs['misc']
                top = ' '.join(bytes(line).decode('latin-1').replace('\0', ' ') for line in obs['tty_chars'][:2])
                msg = (agent.single_message or '') + ' ' + top
                head = msg + ' ' + ' '.join(agent.single_popup[:3])
                single = agent.single_message or ''
                if 'do you want to genocide' in single and 'monster' not in answered:
                    answered.add('monster')
                    for k in 'minotaur\r':
                        yield k
                    continue
                if 'class of monsters do you wish to genocide' in single and 'class' not in answered:
                    answered.add('class')
                    for k in 'H\r':
                        yield k
                    continue
                if 'What would you like to identify' in head:
                    keys = power_route._identify_step(agent, menu_pages)
                    if keys is None:
                        yield A.Command.ESC
                        continue
                    if keys.endswith('\r'):
                        menu_pages.clear()
                    for k in keys:
                        yield k
                    continue
                if 'Where do you want to center' in msg or 'What do you want to charge?' in msg:
                    yield A.Command.ESC
                    continue
                # detect.c browse_map(): gold/food/object detection lets us look around the map with getpos() ('(For
                # instructions type a '?')'); left open, the next command's key went into it ("Unknown direction:
                # 'r' (aborted)", vw-lot1-s7 jf16-s14 and jf26-s0)
                if 'For instructions type a' in msg and 'getpos' not in answered:
                    answered.add('getpos')
                    yield A.Command.ESC
                    continue
                if misc[2] or '--More--' in top:
                    yield ' '
                    continue
                return

        with agent.atom_operation():
            agent.step(A.Command.READ, gen())
        self._log(f'LOTTERY read -> {(agent.message or "")[:200]!r}')
        agent.inventory.items.update(force=True)

    def lottery_strategy(self):
        """Preempt (above the walk): VALLEY_LOTTERY's reads and quaffs. Loops like the walk's body."""
        walker = self

        def f():
            if not walker.lottery_active() or walker.agent.blstats.time < walker._lottery_cooldown:
                yield False
                return
            try:
                action = walker._lottery_plan() if walker._lottery_ok_now() else None
            except Exception as e:   # a planning bug must not break the preempt loop
                walker._log(f'LOTTERY plan failed: {e!r}')
                action = None
            if action is None:
                yield False
                return
            yield True
            agent = walker.agent
            for _ in range(40):
                steps, turn0 = agent.step_count, agent.blstats.time
                walker._lottery_act(action)
                if agent.blstats.time == turn0:
                    walker._lottery_idle += 1
                    if walker._lottery_idle >= 4:
                        walker._lottery_idle = 0
                        walker._lottery_cooldown = agent.blstats.time + 20
                        walker._log('LOTTERY no game time passed in 4 actions: stepping aside 20 turns')
                        return
                else:
                    walker._lottery_idle = 0
                if agent.step_count == steps or not walker.lottery_active() or not walker._lottery_ok_now():
                    return
                action = walker._lottery_plan()
                if action is None:
                    return

        return Strategy(f)

    def strategy(self):
        """Preempt (above fight2, valley_sneak, the Valley retreat/fort, gehennom_escape/scare): every Valley move
        while the walk has a plan. The body loops (a one-action return lets the lower chain take one step)."""
        walker = self

        def f():
            if not walker.active() or walker.agent.blstats.time < walker._cooldown_until:
                yield False
                return
            try:
                action = walker._decide()
            except Exception as e:   # a planning bug must not break the preempt loop
                walker._log(f'decide failed: {e!r}')
                action = None
            if action is None:
                yield False
                return
            yield True
            agent = walker.agent
            for _ in range(200):
                steps, turn0 = agent.step_count, agent.blstats.time
                walker._act(action)
                if agent.blstats.time == turn0:
                    walker._idle += 1
                    if walker._idle >= 10:
                        walker._idle = 0
                        walker._cooldown_until = agent.blstats.time + 5
                        walker._log(f'no game time passed in 10 actions ({action[0]}): stepping aside 5 turns')
                        return
                else:
                    walker._idle = 0
                if agent.step_count == steps or not walker.active():
                    return
                action = walker._decide()
                if action is None:
                    return

        return Strategy(f)
