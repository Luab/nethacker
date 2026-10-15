"""valley-exit's castle landing layer (jf_config.LANDING_GUARD): stay alive the first ~100 turns after the fall onto
the castle (the Dungeons' bottom, Dlvl 25-29), so the crossing routes (a lift, castle-first-pass's rush and xorn,
power-route's teleport-control gamble) get to act.

Harness baseline (castle-real-all, 61 real kits x 4 level-generation salts = 244 games, 3a5b3ef + castle-first-pass):
alive 100 turns after the landing 46%; of the 131 deaths inside those 100 turns 59 were minotaurs -- they share the
west maze in ~42% of castles (mkmaze.c fill_empty_maze: rn2(3) per maze half), arrive a median 6-7 turns after the
fall (infravisible, speed 15) and take 30-50 HP a turn (3d10/3d10/2d8 at to-hit 25+), and Elbereth doesn't stop them
(monmove.c onscary: PM_MINOTAUR). The castle is only recognised after a pit and a second dig (median 6 turns).

What a real kit can do about one (monst.c: minotaur MR 0, no sleep/cold/fire resistance, no regeneration):
  * a known wand at it, best first (castle_power.breach_wand): sleep (zap.c sleep_monst: d(6,25) turns frozen, MR 0
    never resists), teleportation (u_teleport_mon rlocs a monster even on the noteleport castle, F055), polymorph,
    then striking (a beam: no bounce), cold (a ray, but Valkyries are cold resistant: its bounce can't hurt us), and
    fire/lightning/magic missile/sleep rays only with room for the ray to die out before it bounces back at us
    (zap.c dobuzz: range rn1(7,7), a wall bounces it);
  * a frozen (slept) minotaur is struck, not zapped again (cl-mino-all4 jf14-s11 re-zapped sleep at a sleeping one,
    and the bounce put us to sleep beside it);
  * known healing potions while one is next to us at LANDING_HEAL_BELOW of our HP instead of 1/3 (emergency_strategy):
    one of its turns takes 30-50 HP, so 1/3 is often skipped straight to death;
  * a scroll of scare monster: CASTLE_SCARE's drop fires at LANDING_SCARE_RADIUS instead of 2 (dive_logic.gehennom_scare).
It stays out of castle-first-pass's way: never while we float or walk walls (castle_cross), and while their lift
phase still has items to try it only answers a big Elbereth-ignorer next to us.
"""
import re

import nle.nethack as nh

from . import castle_power, jf_config, utils
from . import objects as O
from .item import Item
from .level import Level
from .strategy import Strategy

LANDING_HEAL_BELOW = 0.6
LANDING_SCARE_RADIUS = 4
LANDING_ZAP_RANGE = 8
# a ray travels rn1(7,7) = 7..13 squares (zap.c dobuzz): with this many free squares beyond us along its way it dies
# out before a bounce can bring it back
RAY_SAFE_RUN = 13
# how long a monster hit by our sleep ray counts as frozen (sleep_monst: d(6,25) turns, at least 6)
SLEEP_WINDOW = 12
# a lich within 2 at the castle depth: stand on Elbereth (see strategy step 0)
LANDING_LICH_ELBERETH = True
_LICHES = frozenset(('lich', 'demilich', 'master lich', 'arch-lich'))

_HEAL = ('full healing', 'extra healing', 'healing')
_BEAMS = ('teleportation', 'polymorph', 'striking', 'slow monster')
_SAFE_RAYS = ('cold',)          # Valkyries resist cold (the only role this bot plays)
_RISKY_RAYS = ('sleep', 'fire', 'lightning', 'magic missile')
_SLEEP_HIT = re.compile(r'The sleep ray hits (?:the )?([a-z][a-z -]*?)[.!]')


class LandingGuard:
    def __init__(self, dive):
        self.dive = dive
        self.agent = dive.agent
        self._asleep = {}          # (y, x) -> turn our sleep ray hit a monster there
        self._lich_engraves = 0    # Elbereths written against a lich (a cap: never an engraving loop)
        self._last_msg_turn = -1

    # ------------------------------------------------------------------ state

    def _in_scope(self):
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number != Level.DUNGEONS_OF_DOOM or agent.blstats.depth < jf_config.CASTLE_SCARE_DEPTH:
            return False
        castle = self.dive.castle
        if castle._floating():
            return False
        try:
            from . import castle_cross
            if castle_cross.wallwalker(agent):
                return False
        except Exception:
            pass
        return True

    def _note(self):
        """Remember where our sleep ray froze something (by name: the ray names its victims). Parsed on every call
        (idempotent): a Fast hero's second move in a turn keeps the turn counter, so a turn gate lost the message."""
        agent = self.agent
        turn = agent.blstats.time
        names = _SLEEP_HIT.findall(agent.message or '')
        if not names:
            return
        for _, y, x, mon, _ in agent.get_visible_monsters():
            if getattr(mon, 'mname', '') in names:
                self._asleep[(int(y), int(x))] = turn

    def _big_melee(self, mon):
        """A minotaur, or an @ of level 10+ (captain, Elvenking...): what Elbereth doesn't stop in melee.
        (castle_power.is_big_ignorer also counts casters -- a master lich is in RANGED_MONSTERS -- which made v3's
        lich step fire in 3 of 48 lich games: Elbereth is the answer to a lich, not a zap.)"""
        name = getattr(mon, 'mname', '')
        if name == 'minotaur':
            return True
        return name != 'unknown' and self.dive._melee_ignores_elbereth(mon) and getattr(mon, 'mlevel', 0) >= 10

    def _frozen(self, m):
        t = self._asleep.get((int(m[1]), int(m[2])))
        return t is not None and self.agent.blstats.time - t <= SLEEP_WINDOW

    # ------------------------------------------------------------------ helpers

    def _free_run(self, dy, dx):
        """Walkable squares from us in direction (dy, dx) before a wall or the map's edge."""
        level = self.agent.current_level()
        y, x = int(self.agent.blstats.y), int(self.agent.blstats.x)
        h, w = level.walkable.shape
        n = 0
        while n < 20:
            y, x = y + dy, x + dx
            if not (0 <= y < h and 0 <= x < w) or not level.walkable[y, x]:
                break
            n += 1
        return n

    def _known_wand(self, names):
        agent = self.agent
        for name in names:
            obj = castle_power._W.get(name)
            if obj is None:
                continue
            for it in agent.inventory.items:
                if it.is_wand() and it.is_unambiguous() and it.object == obj and \
                        not castle_power.power._empty(agent, it):
                    return it, name
        return None, None

    def _pick_wand(self, adjacent, dy, dx):
        """The wand to zap at a big ignorer in direction (dy, dx), best first:
        1. known teleportation/polymorph (beams) and sleep (a ray: next to us it freezes the target before any
           bounce can come back; farther only with room for it to die out);
        2. known cold (Valkyries resist its bounce) and fire (next to us: the bounce hits the target twice for the
           ~21 it may cost us); lightning and magic missile only with room (lightning's bounce blinds us);
        3. an unknown wand (castle_power.breach_wand's odds: P(sleep/teleportation/polymorph) ~0.15-0.2 per wand);
        4. known striking / slow monster -- 2d12 a zap can't drop a 15HD minotaur before its 30-50 a turn drops us
           (smoke cl-lg: jf14-s11 zapped striking twice before its unknown wand of sleep and died at 11 turns;
           the baseline, zapping the unknowns, slept it and lived 325)."""
        safe = self._free_run(dy, dx) >= RAY_SAFE_RUN
        wand, name = self._known_wand(('teleportation', 'polymorph'))
        if wand is None and (adjacent or safe):
            wand, name = self._known_wand(('sleep',))
        if wand is None:
            wand, name = self._known_wand(_SAFE_RAYS)
        if wand is None and (adjacent or safe):
            wand, name = self._known_wand(('fire',))
        if wand is None and safe:
            wand, name = self._known_wand(('lightning', 'magic missile'))
        if wand is not None:
            return wand, f'known {name}'
        wand, why = castle_power.breach_wand(self.agent)
        if wand is not None and not wand.is_unambiguous():
            return wand, why
        wand, name = self._known_wand(('striking', 'slow monster'))
        if wand is not None:
            return wand, f'known {name}'
        return None, None

    def _heal_potion(self):
        agent = self.agent
        for name in _HEAL:
            for it in agent.inventory.items:
                if it.category == nh.POTION_CLASS and it.is_unambiguous() and it.object.name == name and \
                        it.status != Item.CURSED:
                    return it
        return None

    # ------------------------------------------------------------------ the strategy

    def strategy(self):
        def f():
            agent = self.agent
            if not jf_config.LANDING_GUARD or not self._in_scope():
                yield False
                return
            self._note()
            if self.dive.on_scare_scroll():
                yield False   # the scare hold answers (a scared minotaur doesn't melee)
                return
            bl = agent.blstats
            pos = (int(bl.y), int(bl.x))
            monsters = agent.get_visible_monsters()
            big = [m for m in monsters if self._big_melee(m[3])]
            # 0. a lich within 2 and no big Elbereth-ignorer next to us: stand on Elbereth. A covetous master/arch-lich
            # of the court teleports next to us (monmove.c tactics: mnexto 1 turn in 5) a median 7 turns after the
            # fall; its touch is cold (Valkyries resist it), the killers are its spells -- psi bolt, touch of death,
            # destroy armor, curse items all come through mattacku, which a scared lich skips (onscary: liches respect
            # Elbereth off Gehennom); only undirected ones (summon nasties, haste, cure) are still cast, and most
            # nasties respect the engraving too. Putting on rings and quaffing don't smudge it; fight2 won't strike a
            # scared monster from it (AT_ELBERETH_FIX).
            if LANDING_LICH_ELBERETH and not any(utils.adjacent((m[1], m[2]), pos) and not self._frozen(m) for m in big):
                liches = [m for m in monsters if getattr(m[3], 'mname', '') in _LICHES and
                          max(abs(int(m[1]) - pos[0]), abs(int(m[2]) - pos[1])) <= 2]
                engraving = (agent.inventory.engraving_below_me or '').lower()
                if liches and engraving != 'elbereth' and agent.can_engrave() and not agent.character.prop.blind \
                        and self._lich_engraves < 12:
                    yield True
                    self._lich_engraves += 1
                    agent.log(f'LANDING Elbereth against the {liches[0][3].mname} at '
                              f'{(int(liches[0][1]), int(liches[0][2]))} hp {bl.hitpoints}/{bl.max_hitpoints}')
                    agent.engrave('Elbereth')
                    return
            if not big:
                yield False
                return
            adjacent = [m for m in big if utils.adjacent((m[1], m[2]), pos)]
            awake_adj = [m for m in adjacent if not self._frozen(m)]
            try:
                from . import castle_cross
                lift_phase = castle_cross.pending(self.dive.castle)
            except Exception:
                lift_phase = False
            # 1. heal early while one is next to us
            if awake_adj and bl.hitpoints < LANDING_HEAL_BELOW * bl.max_hitpoints:
                potion = self._heal_potion()
                if potion is not None:
                    yield True
                    agent.log(f'LANDING quaffing {potion.text!r} at {bl.hitpoints}/{bl.max_hitpoints} beside the '
                              f'{awake_adj[0][3].mname}')
                    agent.inventory.quaff(potion)
                    return
            # 2. strike a frozen one next to us (a zap would only risk a bounce)
            frozen_adj = [m for m in adjacent if self._frozen(m)]
            if frozen_adj and not awake_adj:
                yield True
                m = frozen_adj[0]
                agent.log(f'LANDING striking the frozen {m[3].mname} at {(int(m[1]), int(m[2]))}')
                if not agent.wield_best_melee_weapon():
                    agent.melee_attack(int(m[1]), int(m[2]))
                return
            # 3. zap: at one next to us, or (outside the lift phase) at one in line within range
            targets = [(1, agent.calc_direction(pos[0], pos[1], int(m[1]), int(m[2])), m) for m in awake_adj]
            if not targets and not lift_phase:
                targets = [t for t in castle_power.inline_targets(self.dive, LANDING_ZAP_RANGE)
                           if not self._frozen(t[2])]
            for dist, direction, m in targets:
                dy = int(m[1]) - pos[0]
                dx = int(m[2]) - pos[1]
                dy, dx = (dy > 0) - (dy < 0), (dx > 0) - (dx < 0)
                wand, why = self._pick_wand(dist == 1, dy, dx)
                if wand is None:
                    continue
                yield True
                castle_power.zap_breach_wand(agent, wand, why, m, direction)
                self._note()
                return
            yield False

        return Strategy(f)


def scare_radius():
    """dive_logic.gehennom_scare's castle branch: how close an Elbereth-ignorer comes before the scroll drop."""
    return LANDING_SCARE_RADIUS if jf_config.LANDING_GUARD else 2
