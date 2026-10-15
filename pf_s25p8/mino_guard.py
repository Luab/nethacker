"""minotaur lane (jf_config.MINO_GUARD): keep minotaurs from ending the dive -- in the filler mazes between Medusa and
the castle (mkmaze.c fill_empty_maze: rn2(3) minotaurs per level) and in the castle's west maze.

Evidence (cmp-main = 99e4eb7, 90 pinned games): minotaurs killed 23 of 90 games. Of the 17 on dev sets, 7 died on a
maze level 0-9 turns after the landing (jf14 s0/s9, jf16 s10, jf41 s0/s13, jf42 s5/s7) and 10 on the castle. A
minotaur (monst.c: 15HD, speed 15, 3d10/3d10/2d8 at to-hit 25+, MR 0) takes ~42 HP a turn, so an XL-8 digger (~80 HP)
lives two turns next to one; it ignores Elbereth (monmove.c onscary); and every attack, hit or miss, stops the dig
occupation (mhitu stop_occupation) while the occupation only runs on a move of ours the minotaur doesn't answer, so
the hole (8 dwarf moves: apply, 2 digs to the pit -- which resets the effort -- apply, 4 digs) is out of reach once it
is adjacent. Replays with the MINO kit log (mino-diag-*, byte-identical to cmp-main): 3 of the 7 maze deaths carried
a known item that ends the fight in one action and never used it -- jf14 s0 a scroll of genocide, jf16 s10 two
uncursed scrolls of teleportation (and the '<' one step away), jf41 s13 a wand of teleportation -- and two more an
unknown wand or scroll to gamble on. The minotaur is usually in view first at 2-4 squares (dwarvish infravision,
M3_INFRAVISIBLE), 1-3 turns before its first blow.

While a minotaur is in view within MINO_RANGE (one action per step; best first):
  0. a due emergency prayer (critically low HP, safe) goes first: emergency_strategy (3 invulnerable turns, full HP);
     on a scroll of scare monster (ours) on a diggable level: dig on (onscary: a minotaur respects the scroll, and a
     scared monster doesn't stop the occupation, hack.c monster_nearby);
  1. a known wand of digging zapped down (not on the castle: Can_dig_down is false there);
  2. a known wand at it in a straight line: teleportation / polymorph (beams, no bounce), sleep (a ray: it hits the
     minotaur first; a bounce may put us to sleep too, but its sleep adds up on the way back);
  3. on an up staircase: climb (a minotaur has no M2_STALK: it never follows); that '>' is then avoided a while;
  4. where teleports work (not the castle, Medusa, the Valley, Sokoban): a known wand of teleportation at ourselves,
     a known scroll of teleportation (a cursed one level-teleports -- away too);
  5. a known scroll of genocide (not while confused: that genocides us): 'minotaur' (blessed: the class 'H');
  6. a known scroll of scare monster: drop it and dig from it;
  7. death / sleep-or-death (the engrave test can't tell them apart) in line -- a bounce may come back at us (a wall
     within ~3 squares), but next to a minotaur nothing else left gives better odds;
  8. the up staircase within MINO_STAIRS_STEPS steps;
  9. a known wand of cold (Valkyries resist its bounce), fire or lightning next to us: 6d6 a pass, and a corridor
     bounces the ray through it twice (15d8 = ~67 HP);
 10. an unknown wand at it in line (the best P(sleep/death/teleportation/polymorph), castle_power's odds);
 11. where teleports work and it is within 5: an unknown scroll (P(teleportation + scare monster + taming + genocide)
     is ~15% for an unknown label; the effects that don't help cost only the turn).
And: every minotaur in view frozen by our sleep ray -> dig out now; while one was seen on this level lately the dive
doesn't rest (mino_alert: it hunts us -- monmove.c set_apparxy knows where we are).
"""
import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import castle_power, jf_config, opp_items, power, utils
from . import objects as O
from .glyph import G
from .item import Item
from .level import Level
from .strategy import Strategy

MINO_RANGE = 6            # react to a minotaur in view this close (Chebyshev); beams reach 6-13, rays 7-13
MINO_CONSUME_RANGE = 5    # ...scrolls and self-teleports only this close: at first sight (mino-g1 jf16 s10 saw it at
                          # 4, dug on and read its teleport scroll only after the first 43-HP round)
MINO_STAIRS_STEPS = 2     # walk to an up staircase this many steps away
MINO_STAIRS_AVOID = 1000  # turns the '>' we climbed onto is avoided (dig down elsewhere instead)
MINO_ALERT_TURNS = 300    # after a minotaur sighting on a level, no rest there this long
SLEEP_WINDOW = 150        # our sleep ray froze it for d(6,25) turns (zap.c sleep_monst): frozen until it acts or
                          # moves (_note), at most this long

_SCR = {n: O.from_name(n, nh.SCROLL_CLASS) for n in ('teleportation', 'genocide', 'scare monster', 'taming')}
_W = castle_power._W
_SLEEP_OR_DEATH = frozenset((_W['sleep'], _W['death']))
_SLEEP_HIT = re.compile(r'The sleep ray hits (?:the )?minotaur')
# it is awake: it attacked (mhitu.c hitmsg/missmu) -- each of our blows on a frozen one wakes it 1 time in 10
# (uhitm.c find_roll_to_hit: !mcanmove -> !rn2(10) mcanmove = 1): mh5-on-mino-sleep castle s1/s4 struck the sleeper,
# it woke on the first blow and the guard still counted it frozen for 12 turns (no second zap): dead in 3
_MINO_ACTS = re.compile(r'The minotaur (?:hits|misses|just misses|butts|bites)')
_NOTELEPORT = 'A mysterious force prevents you from teleporting'
GEHENNOM = 1   # dungeon number (dive_logic.GEHENNOM)
_PM_MINOTAUR = next(i for i in range(nh.NUMMONS) if nh.permonst(i).mname == 'minotaur')
# every way a minotaur shows on the map (seen, sensed: telepathy while blind, ridden); the guard does nothing -- not
# even a bfs() call, whose per-step cache other strategies share -- while none of these is on the screen
_MINO_GLYPHS = [off + _PM_MINOTAUR for off in (nh.GLYPH_MON_OFF, nh.GLYPH_PET_OFF, nh.GLYPH_DETECT_OFF,
                                               nh.GLYPH_RIDDEN_OFF)]
_MINO_PERMONST = nh.permonst(_PM_MINOTAUR)


class MinoGuard:
    def __init__(self, dive):
        self.dive = dive
        self.agent = dive.agent
        dive.mino_guard = self
        self._seen = {}            # level key -> last turn a minotaur was in view
        self._asleep = {}          # (level key, (y, x)) -> turn our sleep ray hit a minotaur there
        self._noteleport = set()   # level keys where a teleport was refused
        self._read_glyphs = set()  # unknown scroll labels read by the guard (never twice)
        self._last_note = None
        self._step_blocked_until = -1
        self._blocked = {}         # (kind, item glyph) -> turn until which that action isn't planned
        self._attempts = {}        # (kind, item glyph) -> (turn, tries in that turn)
        self._fled = {}            # HORN_SCARE: level key -> last turn 'The minotaur turns to flee.'
        self._blows = {}           # HORN_SCARE: level key -> turns we blew a scare instrument there
        self._near_prev = {}       # HORN_SCARE: level key -> (step, nearest awake minotaur now, at the step before)
        self._hp_prev = None       # HORN_SCARE: (step, our HP now, at the step before)

    # ------------------------------------------------------------------ state

    def alert(self):
        """A minotaur was in view on this level within MINO_ALERT_TURNS turns: no resting here."""
        if not jf_config.MINO_GUARD:
            return False
        t = self._seen.get(self.agent.current_level().key())
        # (a level we can dig out of: the castle keeps its own rest rules -- no hole to run to there)
        return t is not None and self.agent.blstats.time - t <= MINO_ALERT_TURNS and self._diggable()

    def _in_scope(self):
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, GEHENNOM):
            return False
        prop = agent.character.prop
        if prop.polymorph or prop.hallu:
            return False   # a form's HP is a buffer (and its own plans run: castle_cross); hallucination hides names
        if utils.any_in(agent.glyphs, G.SWALLOW):
            return False
        if self.dive.levitating() or self.dive.castle._floating():
            return False   # over the moat: the crossing's business (a minotaur can't enter water)
        return True

    def _note(self):
        """Idempotent message parsing (runs on every check): our sleep ray froze a minotaur; a refused teleport."""
        agent = self.agent
        msg = agent.message or ''
        key = agent.current_level().key()
        stamp = (agent.step_count, key)
        if self._last_note == stamp:
            return
        self._last_note = stamp
        if _NOTELEPORT in msg:
            self._noteleport.add(key)
        if _MINO_ACTS.search(msg):
            # awake (the message of this step: a sleep ray's hit in the same message comes after its blows)
            last = max(msg.rfind('The minotaur hits'), msg.rfind('The minotaur misses'),
                       msg.rfind('The minotaur just misses'), msg.rfind('The minotaur butts'),
                       msg.rfind('The minotaur bites'))
            ray = msg.rfind('The sleep ray hits')
            if ray < last:
                self._asleep = {k: v for k, v in self._asleep.items() if k[0] != key}
        if _SLEEP_HIT.search(msg):
            for m in self._minos():
                self._asleep[(key, (m[1], m[2]))] = agent.blstats.time
        if jf_config.HORN_SCARE and 'The minotaur turns to flee' in msg:
            self._fled[key] = agent.blstats.time

    def _minos(self):
        """Minotaurs on the screen: [(chebyshev distance, y, x, permonst)], nearest first. From the glyphs, not
        agent.get_visible_monsters(): that one keeps only monsters next to a square our BFS reaches, and in a dark
        maze the squares around a minotaur seen by infravision are unknown -- jf16 s10 (mino-g5) saw it 4 squares
        off on the landing turn, the guard only once it hit (a monster in view has a clear line: vision.c)."""
        agent = self.agent
        bl = agent.blstats
        ys, xs = utils.isin(agent.glyphs, _MINO_GLYPHS).nonzero()
        out = []
        for y, x in zip(ys, xs):
            y, x = int(y), int(x)
            if (y, x) == (int(bl.y), int(bl.x)):
                continue
            out.append((max(abs(y - int(bl.y)), abs(x - int(bl.x))), y, x, _MINO_PERMONST))
        out.sort(key=lambda m: m[0])
        return out

    def _frozen(self, m):
        t = self._asleep.get((self.agent.current_level().key(), (m[1], m[2])))
        return t is not None and self.agent.blstats.time - t <= SLEEP_WINDOW

    def _diggable(self):
        """This level can be holed (not the castle or another bottom, not the Valley's hardfloor)."""
        dive = self.dive
        level = self.agent.current_level()
        key = level.key()
        if key in dive.undiggable or key == dive.castle.castle_key or dive.in_valley() or \
                level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, GEHENNOM):
            return False
        return True

    def _teleport_ok(self):
        """Teleporting within this level works: filler mazes yes; the castle, Medusa's island, the Valley and the
        other noteleport levels no (castle.des/medusa.des/gehennom.des FLAGS: noteleport)."""
        dive = self.dive
        level = self.agent.current_level()
        key = level.key()
        if key in self._noteleport or key == dive.castle.castle_key or key == dive.medusa_level or \
                dive.in_valley() or level.dungeon_number == Level.SOKOBAN or self._maybe_castle():
            return False
        return True

    def _maybe_castle(self):
        """This may be the castle, not yet recognised (that takes a failed dig): the Dungeons at castle depth
        (CASTLE_SCARE_DEPTH, 25+), no castle known elsewhere, and we arrived inside its fall region (castle.des
        levregion x 1-10 = bot x 0-9; a filler maze lands us at a random spot). Harness mh4-on-mino-telescroll: the
        guard's teleport scroll was refused on all 6 castle landings ('A mysterious force prevents you from
        teleporting!') and those turns went to the minotaur."""
        dive = self.dive
        agent = self.agent
        level = agent.current_level()
        key = level.key()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or agent.blstats.depth < jf_config.CASTLE_SCARE_DEPTH:
            return False
        castle = dive.castle.castle_key
        if castle is not None:
            return castle == key
        vkey, vpos = getattr(dive, '_visit_pos', (None, None))
        return vkey == key and vpos is not None and int(vpos[1]) <= 9

    def _prayer_first(self):
        """emergency_strategy (below us) would pray now: let it -- the prayer's 3 invulnerable turns and full HP
        come before any item (then the item from full HP)."""
        agent = self.agent
        bl = agent.blstats
        if agent.prayer_failed:
            return False
        xl = bl.experience_level
        low = bl.hitpoints < bl.max_hitpoints / (5 if xl < 6 else 6) or \
            (bl.hitpoints < 12 and bl.hitpoints < bl.max_hitpoints)
        if low and agent.is_safe_to_pray(500):
            return True
        return bool(jf_config.DESPERATE_PRAYER_GAP) and agent._critically_low_hp() and \
            agent.current_level().dungeon_number != GEHENNOM and self.dive.diving and bl.depth >= 5 and \
            agent.is_safe_to_pray(jf_config.DESPERATE_PRAYER_GAP)

    def reserved(self, item):
        """fight2 (fight_heur.get_potential_wand_usages) must not zap this at a monster: a known wand of death -- the
        one ray wand fight2 fires that stops a minotaur for good -- is kept for one, except in an HP emergency (below
        WAND_RESERVE_HP with a hostile within 2). (fight2 never zaps sleep, teleportation or polymorph: beams and
        sleep aren't 'offensive usable'; cold is COLD_RESERVE's.)"""
        if not jf_config.MINO_GUARD or item.category != nh.WAND_CLASS or not item.is_unambiguous() or \
                item.object != _W['death']:
            return False
        return not self.dive._reserve_emergency()

    # ------------------------------------------------------------------ items

    def _wand(self, names):
        """A known, not-empty wand of one of `names` (in that order): (item, name) or (None, None)."""
        agent = self.agent
        for name in names:
            obj = _W[name]
            for it in agent.inventory.items:
                if it.is_wand() and it.is_unambiguous() and it.object == obj and not power._empty(agent, it):
                    return it, name
        return None, None

    def _sleep_or_death(self):
        agent = self.agent
        for it in agent.inventory.items:
            if it.is_wand() and not it.is_unambiguous() and set(it.objs) <= _SLEEP_OR_DEATH and \
                    not power._empty(agent, it):
                return it
        return None

    def _scroll(self, name, cursed_ok=False):
        agent = self.agent
        for it in agent.inventory.items:
            if it.category == nh.SCROLL_CLASS and it.is_unambiguous() and it.object == _SCR[name] and \
                    (cursed_ok or it.status != Item.CURSED):
                if jf_config.GENOCIDE_POLICY and name == 'genocide' and opp_items.proven_cursed(self.agent, it):
                    continue   # its stack's first read sent monsters in: cursed (4-6 more minotaurs)
                return it
        return None

    def _unknown_wand(self, sy, sx):
        """The unknown wand (not zapped yet) with the best castle_power.zap_value at a minotaur in direction (sy, sx):
        what each type it may still be does to it (MR 0, no resistances: teleportation/sleep/death end it, polymorph
        mostly, cold/fire/lightning hurt it) minus the helping types (speed monster, make invisible, create monster)
        and a ray's bounce onto us (sleep, death: the free line behind it). (item, why) or (None, None)."""
        agent = self.agent
        best, why = castle_power.best_zap(agent, _MINO_PERMONST, sy, sx, unknown=True)
        if best is None or best.is_unambiguous():
            # (a known wand here: the steps above chose not to use it -- no gamble with it)
            known = {it.glyphs[0] for it in agent.inventory.items if it.is_wand() and it.is_unambiguous()}
            best, why = castle_power.best_zap(agent, _MINO_PERMONST, sy, sx, unknown=True, exclude=known)
        if best is None or best.is_unambiguous():
            return None, None
        return best, why

    def _unknown_scroll(self):
        """The unknown scroll most likely to end the fight: P(teleportation) (where teleports work) + P(scare monster)
        + P(taming) + P(genocide); (item, p) or (None, 0)."""
        agent = self.agent
        good = {_SCR['scare monster'], _SCR['taming'], _SCR['genocide']}
        if self._teleport_ok():
            good.add(_SCR['teleportation'])
        best, bp = None, 0.0
        for it in agent.inventory.items:
            if it.category != nh.SCROLL_CLASS or it.is_unambiguous() or it.glyphs[0] in self._read_glyphs:
                continue
            p = power.p_of(it, good)
            if p > bp:
                best, bp = it, p
        return best, bp

    # ------------------------------------------------------------------ geometry

    def _line(self, m):
        """(direction, distance) toward minotaur m if it is in a straight line within MINO_RANGE, else None."""
        bl = self.agent.blstats
        dy, dx = m[1] - int(bl.y), m[2] - int(bl.x)
        dist = max(abs(dy), abs(dx))
        if dist == 0 or dist > MINO_RANGE or not (dy == 0 or dx == 0 or abs(dy) == abs(dx)):
            return None
        sy, sx = (dy > 0) - (dy < 0), (dx > 0) - (dx < 0)
        return self.agent.calc_direction(bl.y, bl.x, bl.y + sy, bl.x + sx), dist, (sy, sx)

    def _free_run(self, sy, sx):
        """Walkable squares from us in direction (sy, sx) before a wall or the map's edge (unseen squares stop it)."""
        level = self.agent.current_level()
        y, x = int(self.agent.blstats.y), int(self.agent.blstats.x)
        h, w = level.walkable.shape
        n = 0
        while n < 20:
            y, x = y + sy, x + sx
            if not (0 <= y < h and 0 <= x < w) or not level.walkable[y, x]:
                break
            n += 1
        return n

    def _ups(self):
        """Up staircases on this level we could climb to escape: [(steps, (y, x))] (the stairs we came down by show
        us, not the stairs: their memory and the dive's arrival square know them)."""
        agent = self.agent
        dive = self.dive
        level = agent.current_level()
        if level.dungeon_number == Level.SOKOBAN or agent.blstats.depth <= 1 or dive.in_valley():
            return []
        if jf_config.MEDUSA_NO_RETREAT and dive._up_leads_to_medusa():
            return []   # onto Medusa's '>', next to her (stoning)
        ups = {(int(y), int(x)) for y, x in zip(*utils.isin(level.objects, G.STAIR_UP).nonzero())}
        ups |= {(int(p[0]), int(p[1])) for p, dest in level.stair_destination.items()
                if dest[0][0] == level.dungeon_number and dest[0][1] < level.level_number}
        arr = getattr(dive, '_arrival_square', None)
        if arr is not None and arr[0] == level.key():
            ups.add((int(arr[1][0]), int(arr[1][1])))
        pos = (int(agent.blstats.y), int(agent.blstats.x))
        if pos in ups:
            return [(0, pos)]
        dis = agent.bfs()
        return sorted((int(dis[p]), p) for p in ups if dis[p] > 0)

    # ------------------------------------------------------------------ the plan

    def _plan(self):
        """The guard's next action (kind, arg, why) or None: the first candidate not blocked by the loop guard.
        Side-effect free apart from bookkeeping: it runs as the preempt condition on every step."""
        if not jf_config.MINO_GUARD or not utils.isin(self.agent.glyphs, _MINO_GLYPHS).any() or \
                not self._in_scope():
            return None
        turn = self.agent.blstats.time
        for plan in self._candidates():
            if self._blocked.get(self._block_key(plan), -1) < turn:
                return plan
        return None

    @staticmethod
    def _block_key(plan):
        kind, arg, _ = plan
        item = arg[0] if isinstance(arg, tuple) and arg and hasattr(arg[0], 'glyphs') else arg
        return (kind, item.glyphs[0] if hasattr(item, 'glyphs') else None)

    def _candidates(self):
        """The guard's actions (kind, arg, why), best first (see the module docstring)."""
        agent = self.agent
        dive = self.dive
        self._note()
        minos = self._minos()
        if not minos:
            return
        bl = agent.blstats
        key = agent.current_level().key()
        self._seen[key] = bl.time
        awake = [m for m in minos if not self._frozen(m)]
        diggable = self._diggable()
        if not awake:
            # every minotaur in view is frozen by our sleep ray: out now (a frozen monster neither attacks nor
            # stops the occupation: hack.c monster_nearby !mcanmove)
            if diggable and minos[0][0] <= MINO_RANGE and self._dig_action() is not None:
                yield ('dig', None, f'minotaur asleep at {minos[0][0]}')
            return
        near = awake[0][0]
        if jf_config.HORN_SCARE:
            # (the distance and our HP at the last earlier step: (step, now, before))
            prev = self._near_prev.get(key)
            if prev is None or prev[0] != agent.step_count:
                self._near_prev[key] = (agent.step_count, near, prev[1] if prev is not None else None)
            hp = self._hp_prev
            if hp is None or hp[0] != agent.step_count:
                self._hp_prev = (agent.step_count, int(bl.hitpoints), hp[1] if hp is not None else None)
        if near > MINO_RANGE:
            return
        if self._prayer_first():
            return
        if dive.on_scare_scroll():
            if diggable and self._dig_action() is not None:
                yield ('dig', None, 'on a scroll of scare monster')
            return   # (the castle's scare hold: gehennom_scare)
        adjacent = near <= 1
        consume = near <= MINO_CONSUME_RANGE or adjacent
        # 1. a known wand of digging down
        if diggable:
            wand = dive._dig_wand()
            if wand is not None and dive._wand_escape(wand) is not None:
                yield ('zapdown', wand, 'known wand of digging')
        # 1b. HORN_SCARE (opp_items): a scare instrument -- a tooled horn, any drum, a horn that scared before -- makes
        # every minotaur in its range flee with no timer (music.c awaken_monsters: MR 0 never resists); one that can
        # move away doesn't attack (monmove.c dochug), so while it runs the dive digs on and nothing is spent on it.
        # Blown again when it attacks (cornered, or the flee ended); not with a minotaur frozen by our sleep ray in
        # view (the sound wakes it)
        kit = self._instruments() if jf_config.HORN_SCARE else {}
        if kit and not any(self._frozen(m) for m in minos):
            # it hit us: its message, or HP lost since the last step with it next to us (the message is gone by the
            # time the guard looks after a --More-- or a zero-time action: oh1on mino-camera s7 dug on beside a
            # blinded, 'fleeing' minotaur from 80 to 19 HP in one turn)
            hp = self._hp_prev
            acted = bool(_MINO_ACTS.search(agent.message or '')) or \
                (near <= 1 and hp is not None and hp[2] is not None and hp[1] < hp[2] - 3)
            fled = self._fled.get(key)   # the last 'turns to flee' or scare blow here
            scared = fled is not None and bl.time - fled <= jf_config.HORN_REFRESH
            # came back next to us since an earlier step: that flee is over (an untimed one ends 1 time in 25 per
            # move at full HP: harness oi-smk mino-horn s0 was back after 10 turns)
            ent = self._near_prev.get(key)
            approaching = near <= 1 and ent is not None and ent[2] is not None and ent[2] >= 2
            recent = [t for t in self._blows.get(key, []) if bl.time - t <= 2]
            d2 = self._dist2(awake[0])
            scare = next((it for it, r2 in kit.get('scare', []) if d2 < r2), None)
            can_blow = scare is not None and len(recent) < 2
            if can_blow and (not scared or approaching):
                yield ('horn', (scare, self._ray_dir(awake)), f'scare instrument, minotaur at {near} (d2 {d2})')
            if scared and (kit.get('scare') or kit.get('camera')):
                # it runs from us: dig now -- not fight2's swing (oi-smk mino-camera s0 hit the fleeing, blinded
                # minotaur and died in 2 turns) nor the dive's Elbereth (minotaurs ignore it; the dive fights an
                # Elbereth-ignorer within 2 squares instead of digging). Still next to us: a fleeing monster with a
                # free square always moves (monmove.c m_move takes the first candidate), so this one is cornered
                # -- in a 1-wide maze dead end it attacks from there every turn (oi-smk2 drum s3: 80 -> 30 HP while
                # digging beside it): step to a square out of its reach first
                if near <= 1 and not dive._in_own_pit() and not agent.in_pit():
                    spot = self._step_away(awake)
                    if spot is not None:
                        yield ('step_away', spot, f'the scared minotaur is next to us: out of its reach to {spot}')
                if can_blow and (acted or approaching):
                    yield ('horn', (scare, self._ray_dir(awake)), f'scare instrument again, minotaur at {near}')
                if diggable and self._dig_action() is not None and not (near <= 1 and acted):
                    yield ('dig', None, f'the minotaur flees (at {near})')
                if not (near <= 1 and acted):
                    return
        # 2. a known wand at it in line: teleportation, polymorph, sleep (we keep our square and its pit)
        lines = [(self._line(m), m) for m in awake]
        lines = [(ln, m) for ln, m in lines if ln is not None]
        if lines:
            (direction, dist, (sy, sx)), m = lines[0]
            wand, name = self._wand(('teleportation', 'polymorph', 'sleep'))
            if wand is not None:
                yield ('zap', (wand, direction, m), f'known {name} at {dist}')
        # 2b. HORN_SCARE: an expensive camera in line within 2 squares (uhitm.c flash_hits_mon: dist2 < 9 -> flees 3
        # times in 4; blinded, for good when adjacent)
        if kit.get('camera') and lines:
            (direction, dist, (sy, sx)), m = lines[0]
            if dist <= 2:
                yield ('camera', (kit['camera'][0], direction, m), f'camera at {dist}')
        # 3. on an up staircase
        ups = self._ups()
        if ups and ups[0][0] == 0 and near <= MINO_CONSUME_RANGE:
            yield ('climb', None, 'on the up stairs')
        teleport = self._teleport_ok()
        # 4. teleport ourselves away (a known wand at us, a known scroll)
        if consume and teleport:
            wand, _ = self._wand(('teleportation',))
            if wand is not None:
                yield ('zapself', wand, 'known teleportation at ourselves')
            scroll = self._scroll('teleportation')
            if scroll is not None and not agent.character.prop.blind:
                yield ('read', scroll, 'known teleportation')
        # 5. genocide
        if consume and not agent.character.prop.confusion and not agent.character.prop.blind:
            scroll = self._scroll('genocide')
            if scroll is not None:
                yield ('read', scroll, 'known genocide')
        # 6. scare monster: drop it and dig from it
        if consume and diggable:
            scroll = self._scroll('scare monster', cursed_ok=True)
            if scroll is not None and agent.current_level().objects[bl.y, bl.x] not in G.STAIR_UP and \
                    agent.current_level().objects[bl.y, bl.x] not in G.STAIR_DOWN:
                yield ('drop_scare', scroll, 'known scare monster')
        # 7. death or sleep-or-death in line. A bounce can come back at us (zap.c dobuzz: range rn1(7,7), -2 per monster
        # hit, a wall reverses a straight ray): out and back past an adjacent minotaur costs ~2f+6 of it with f free
        # squares beyond us, so a wall within ~3 squares may bring it home (zap_hit(u.uac) ~75%). Still the best odds
        # left once nothing above applies -- next to a minotaur we live ~2 turns -- so no geometry rule (the map
        # doesn't know the dark corridor behind it anyway)
        if lines:
            (direction, dist, (sy, sx)), m = lines[0]
            wand, name = self._wand(('death',))
            if wand is None:
                wand, name = self._sleep_or_death(), 'sleep or death'
            if wand is not None:
                yield ('zap', (wand, direction, m), f'known {name} at {dist}')
        # 8. an up staircase a step or two away (not out of our pit: climbing out takes turns)
        if ups and 0 < ups[0][0] <= MINO_STAIRS_STEPS and consume and bl.time >= self._step_blocked_until and \
                not agent.in_pit() and (not adjacent or bl.hitpoints >= 0.5 * bl.max_hitpoints):
            yield ('step_up', ups[0][1], f'up stairs {ups[0][0]} steps away')
        # 8b. HORN_SCARE: an unknown horn within a tooled horn's scare range -- tooled 5/11 (it flees), frost/fire
        # 4/11 (a 6d6 ray: at it when in line, else down the longest free line), plenty 2/11 (nothing)
        if kit.get('horn') and consume and self._dist2(awake[0]) < 10 * int(bl.experience_level):
            yield ('horn', (kit['horn'][0], self._ray_dir(awake)), f'unknown horn, minotaur at {near}')
        # 9. damage rays: cold (a Valkyrie resists its bounce), fire and lightning next to us
        if lines:
            (direction, dist, (sy, sx)), m = lines[0]
            wand, name = self._wand(('cold',))
            if wand is None and dist <= 1:
                wand, name = self._wand(('fire', 'lightning'))
            if wand is not None:
                yield ('zap', (wand, direction, m), f'known {name} at {dist}')
        # 10. an unknown wand at it
        if lines and consume:
            (direction, dist, (sy, sx)), m = lines[0]
            wand, why = self._unknown_wand(sy, sx)
            if wand is not None:
                yield ('zap', (wand, direction, m), f'unknown wand ({why}) at {dist}')
        # 11. an unknown scroll where a teleport would work -- at first sight: in view (a straight corridor) it is 1-4
        # turns off, the hole 5 (from our pit) to 8 moves: that race is lost, so the turns before its first blow are
        # the cheapest ones to gamble with (next to us each one costs ~42 HP)
        if teleport and consume and not agent.character.prop.blind and not agent.character.prop.confusion:
            scroll, p = self._unknown_scroll()
            if scroll is not None and p > 0:
                yield ('read', scroll, f'unknown scroll P(save)={p:.2f}')
        return

    def _instruments(self):
        """HORN_SCARE: {'scare': [(item, scare radius^2)], 'horn': [unknown horns], 'camera': [cameras]}."""
        out = {}
        for it in self.agent.inventory.items:
            kind = opp_items.instrument_kind(self.agent, it)
            if kind == 'scare':
                out.setdefault('scare', []).append((it, opp_items.scare_radius2(self.agent, it)))
            elif kind is not None:
                out.setdefault(kind, []).append(it)
        return out

    def _step_away(self, awake):
        """HORN_SCARE: a square one move away that no awake minotaur is next to (a diggable one first), or None."""
        agent = self.agent
        level = agent.current_level()
        bl = agent.blstats
        dis = agent.bfs()
        h, w = level.walkable.shape
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = int(bl.y) + dy, int(bl.x) + dx
                if (dy, dx) == (0, 0) or not (0 <= y < h and 0 <= x < w) or dis[y, x] != 1 or \
                        agent.monster_tracker.monster_mask[y, x]:
                    continue
                if any(max(abs(y - int(m[1])), abs(x - int(m[2]))) <= 1 for m in awake):
                    continue
                far = min(max(abs(y - int(m[1])), abs(x - int(m[2]))) for m in awake)
                key = (bool(self.dive._diggable_spot(y, x)), far)
                if best is None or key > best[0]:
                    best = (key, (y, x))
        return None if best is None else best[1]

    def _dist2(self, m):
        bl = self.agent.blstats
        return (int(m[1]) - int(bl.y)) ** 2 + (int(m[2]) - int(bl.x)) ** 2

    def _ray_dir(self, awake):
        """Where a possible horn ray goes: at the nearest minotaur in line, else down the longest free line."""
        for m in awake:
            ln = self._line(m)
            if ln is not None:
                return ln[0]
        return opp_items.longest_free_dir(self.agent)[0]

    def _dig_action(self):
        """What 'dig out now' would do: ('pick', tool) / ('wand', wand) / None."""
        dive = self.dive
        agent = self.agent
        bl = agent.blstats
        wand = dive._dig_wand()
        if wand is not None and dive._wand_escape(wand) is not None:
            return ('wand', wand)
        tool = dive.digging_tool()
        if tool is None or agent.blstats.time < dive._dig_blocked_until:
            return None
        if tool.object == O.from_name('dwarvish mattock') and agent.inventory.items.off_hand is not None:
            return None
        max_wet = dive._dig_max_wet() or 0
        on_stairs = (bl.y, bl.x) in agent.current_level().stair_destination
        if dive._in_own_pit() or (not on_stairs and dive._diggable_spot(bl.y, bl.x, max_wet)):
            return ('pick', tool)
        return None

    # ------------------------------------------------------------------ acting

    def _act(self, plan):
        agent = self.agent
        dive = self.dive
        kind, arg, why = plan
        bkey = self._block_key(plan)
        t, n = self._attempts.get(bkey, (None, 0))
        n = n + 1 if t == agent.blstats.time else 1
        self._attempts[bkey] = (agent.blstats.time, n)
        if n >= 3:
            # the same action a third time within one game turn: it takes no time (refused) -- don't loop on it
            self._blocked[bkey] = agent.blstats.time + 3
            agent.log(f'MINO {bkey} refused 3 times at T{agent.blstats.time}: blocked for 3 turns')
        bl = agent.blstats
        hp = f'hp {bl.hitpoints}/{bl.max_hitpoints}'
        key = agent.current_level().key()
        if kind == 'dig':
            what = self._dig_action()
            agent.log(f'MINO dig out ({why}), {hp}')
            if what is None:
                return
            if what[0] == 'wand':
                dive._escape_act(('zap', what[1]))
            else:
                dive.dig_with_tool(what[1])
            return
        if kind == 'zapdown':
            agent.log(f'MINO zapping {arg.text!r} down ({why}), {hp}')
            dive._escape_act(('zap', arg))
            return
        if kind == 'climb':
            agent.log(f'MINO up the stairs ({why}), {hp}')
            agent.move('<')
            if agent.current_level().key() != key:
                # don't come straight back down onto it: dig down elsewhere (a hole lands at a random spot)
                dive._avoid_stairs_until[(agent.current_level().key(), (agent.blstats.y, agent.blstats.x))] = \
                    agent.blstats.time + MINO_STAIRS_AVOID
            return
        if kind == 'step_up':
            agent.log(f'MINO to the up stairs at {arg} ({why}), {hp}')
            start = (bl.y, bl.x)
            try:
                agent.go_to(arg[0], arg[1], max_steps=1)
            finally:
                if (agent.blstats.y, agent.blstats.x) == start:
                    self._step_blocked_until = agent.blstats.time + 5   # a monster in the way: no step loop
            return
        if kind == 'zap':
            wand, direction, m = arg
            if not wand.is_unambiguous():
                agent._last_resort_zapped.add(wand.glyphs[0])
            agent.log(f'MINO zapping {wand.text!r} {direction} at the minotaur at {(m[1], m[2])} ({why}), {hp}')
            agent.zap(wand, direction)
            agent.log(f'MINO zap -> {(agent.message or "")[:160]!r}')
            if 'Nothing happens' in (agent.message or '') or 'You wrest' in (agent.message or ''):
                agent.inventory.empty_wands.add(wand.text)
            self._note()
            agent.inventory.items.update(force=True)
            return
        if kind == 'zapself':
            agent.log(f'MINO zapping {arg.text!r} at ourselves ({why}), {hp}')
            agent.zap(arg, '.')
            agent.log(f'MINO zap -> {(agent.message or "")[:160]!r}')
            self._note()
            agent.inventory.items.update(force=True)
            return
        if kind == 'read':
            self._read(arg, why)
            return
        if kind == 'horn':
            item, ray = arg
            agent.log(f'MINO blowing {item.text!r} ({why}), {hp}')
            self._blows.setdefault(key, []).append(bl.time)
            res = opp_items.play(agent, item, ray)
            if any(snd in res['msg'] for snd in opp_items.SCARE_SOUNDS):
                # MR 0: every minotaur within range flees (an already fleeing one prints nothing)
                self._fled[key] = agent.blstats.time
            self._note()
            return
        if kind == 'step_away':
            agent.log(f'MINO stepping to {arg} ({why}), {hp}')
            start = (bl.y, bl.x)
            try:
                agent.go_to(arg[0], arg[1], max_steps=1)
            finally:
                if (agent.blstats.y, agent.blstats.x) == start:
                    self._blocked[bkey] = agent.blstats.time + 2   # the step failed: no loop on it
            return
        if kind == 'camera':
            item, direction, m = arg
            agent.log(f'MINO flashing {item.text!r} {direction} at the minotaur at {(m[1], m[2])} ({why}), {hp}')
            opp_items.use_camera(agent, item, direction)
            # a blinded monster resists the next flash (resists_blnd): not again at it for a while
            self._blocked[bkey] = agent.blstats.time + 30
            self._note()
            return
        if kind == 'drop_scare':
            agent.log(f'MINO dropping {arg.text!r} ({why}), {hp}')
            pos = (bl.y, bl.x)
            dive._scare_spot = (key, pos)
            dive._scare_drop_turn = bl.time
            agent.inventory.drop(arg, 1)
            agent.inventory._note_dropped([arg], [1], force=True)
            return
        raise ValueError(kind)

    def _read(self, item, why):
        """Read a scroll, answering what it may ask: genocide ('minotaur'; a blessed one's class 'H'), the identify
        menu (power_route's picks), a stinking cloud's aim and a charging scroll (declined)."""
        agent = self.agent
        from . import power_route
        if not item.is_unambiguous():
            self._read_glyphs.add(item.glyphs[0])
        opp_items.note_read(agent, item, why)
        letter = agent.inventory.items.get_letter(item)
        bl = agent.blstats
        agent.log(f'MINO reading {item.text!r} ({letter}): {why}, hp {bl.hitpoints}/{bl.max_hitpoints}')
        menu_pages = {}
        answered = set()

        def gen():
            if 'What do you want to read?' not in agent.single_message:
                return
            yield letter
            for _ in range(60):
                obs = agent._observation
                misc = obs['misc']
                # the screen too: 'As you read the scroll, it disappears.--More--' before the genocide prompt came with
                # misc[2] unset (mino-g1-jf14 s0: this loop ended there and the default handler ESCaped the prompt)
                top = ' '.join(bytes(line).decode('latin-1').replace('\0', ' ') for line in obs['tty_chars'][:2])
                msg = (agent.single_message or '') + ' ' + top
                head = msg + ' ' + ' '.join(agent.single_popup[:3])
                if len(answered) < 2:
                    more = any(b'--More--' in bytes(line) for line in obs['tty_chars'])
                    agent.log(f'MINO read screen: misc={[int(v) for v in misc]} more={more} '
                              f'single={agent.single_message[:90]!r} top={" ".join(top.split())[:120]!r}')
                # (the prompt shown now, as agent.engrave answers its getlin: misc[1] didn't flag this one --
                # mino-g2-jf14 s0 left it for the default handler twice)
                single = agent.single_message or ''
                if 'do you want to genocide' in single and 'monster' not in answered:
                    answered.add('monster')
                    # GENOCIDE_POLICY: 'minotaur' too with one this close, unless the stack proved cursed
                    text = opp_items.genocide_answer(agent, False)[0] if jf_config.GENOCIDE_POLICY else 'minotaur'
                    agent.log(f'MINO genocide answer {text!r}')
                    for k in text + '\r':
                        yield k
                    continue
                if 'class of monsters do you wish to genocide' in single and 'class' not in answered:
                    answered.add('class')
                    text = opp_items.genocide_answer(agent, True)[0] if jf_config.GENOCIDE_POLICY else 'H'
                    for k in text + '\r':
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
                # (NLE flags in_getlin already while the message before the genocide prompt waits on its --More--:
                # misc = [0, 1, 1] -- dismiss it regardless of misc[1])
                if misc[2] or '--More--' in top:
                    yield ' '
                    continue
                return

        with agent.atom_operation():
            agent.step(A.Command.READ, gen())
        agent.log(f'MINO read -> {(agent.message or "")[:200]!r}')
        self._note()
        agent.inventory.items.update(force=True)

    def strategy(self):
        def f():
            plan = self._plan()
            if plan is None:
                yield False
                return
            yield True
            self._act(plan)
            if not (jf_config.HORN_SCARE and self._instruments()):
                return
            # HORN_SCARE with an instrument in the pack: keep acting while the guard has a plan (as HOLD_LOOP). After a
            # one-action return agent.preempt runs one step of the lower chain before this condition is checked
            # again: fight2 swung at the fleeing minotaur, the dive engraved Elbereth (useless against it) between
            # the horn and the dig (harness oi-smk mino-horn s0, mino-camera s0)
            for _ in range(20):
                steps = self.agent.step_count
                plan = self._plan()
                if plan is None:
                    return
                self._act(plan)
                if self.agent.step_count == steps:
                    return

        return Strategy(f)
