"""dive-audit lane (jf_config.KNOWN_ITEMS): in mortal danger while diving, use the KNOWN item that ends the danger
before emergency_strategy's last resort gambles on unknown ones.

Evidence (cand-g = s24 = 6ec6675, 270 fresh pinned games jf43-60; ledger F092). 77 of the 171 deaths with an
inventory log (prep-metric's PREP_LOG twins, jf43-54) held a known escape or attack item they never used: ~30 scrolls
of teleportation, ~15 wands of striking, wands of sleep/fire/cold/lightning, healing potions. In the Dlvl 5-20 band:
  * jf45 s14: a mumak on Mines 4 took 79 -> 43 -> 15 -> dead; the pack held a known wand of sleep and a blessed scroll
    of teleportation;
  * jf52 s9 / jf54 s14: blinded by a yellow light, then a soldier ant (75 -> 0 in 7 turns) / killer bees; 2 uncursed
    scrolls of teleportation / a scroll of teleportation and wands of cold and fire;
  * jf49 s7: stuck to a giant mimic on Dlvl 20 with a scroll of teleportation; jf48 s8: a soldier on Dlvl 10, the same.
Why they sat unused: the last resort (agent.emergency_strategy) tries only UNKNOWN wands, potions and scrolls; fight2
zaps only ray wands (item.is_offensive_usable_wand: never striking, a beam, never sleep); nothing reads a known scroll
of teleportation or quaffs a known healing potion in a fight; and the last resort waits for pray.c's critically low HP
(1/6 of max at XL 6-13), which a 20-40-a-turn attacker skips straight over (jf45 s14: 43 -> 15 -> dead).

While diving (the Dungeons, the Mines, Gehennom), not polymorphed, engulfed or afloat, with a hostile next to us -- or
hurt this badly with one within 7 -- and either
  (a) critically low HP (agent._critically_low_hp, the last resort's trigger), or
  (b) a burst: the HP lost over the last KNOWN_ITEMS_BURST_TURNS turns is at least what is left, below
      KNOWN_ITEMS_BURST_FRAC of max HP (the next such exchange kills us),
and no safe emergency prayer due (that goes first: mino_guard.MinoGuard._prayer_first), the guard takes the first of:
  1. the up staircase we stand on, unless an M2_STALK monster next to us would follow (monst.c: trolls, soldiers...);
  2. a known scroll of teleportation where teleports work (MinoGuard._teleport_ok: not the castle, Medusa's level,
     the Valley, Sokoban -- their .des FLAGS: noteleport -- nor an unrecognised level at Medusa's depth with water) --
     never a known-cursed one or WISH_TELEPORT_ROUTE's stack (power_route's level-teleport tickets), never with
     teleport control known (the scroll would ask where). Readable blind: read.c doread refuses a blind read only of a
     scroll never seen (dknown);
  3. a known wand of teleportation at ourselves (the same levels);
  4. a known wand of digging down, unless an M2_STALK monster next to us would follow through the hole (jf47 s10's
     ice troll followed three zapped holes);
  5. a known potion of full or extra healing (plain healing only at critically low HP: 6d4);
  6. at the strongest attacker next to us: a known wand of sleep (or the engrave-tested 'sleep or death' pair, at
     critically low HP), a ray -- only with KNOWN_ITEMS_RAY_RUN free squares behind it or at critically low HP; a known
     wand of striking (a beam: no bounce); a known wand of cold (Valkyries resist its bounce); fire next to us;
     lightning / magic missile only with room for the ray to die out.
No zap in Minetown or with a peaceful (a watchman, a shopkeeper) on the line (fight_heur.missiles_risk_the_watch), no
teleport with unpaid shop goods in the pack (theft). No danger while an intact Elbereth under us holds everything next
to us and nothing has hurt us through it, and never a zap at a monster that Elbereth scares from it (mon.c setmangry:
'You feel like a hypocrite', -5 alignment, the engraving deleted -- da-all2 jf54 s2 did that to a mumak). Every action is logged 'KNOWN_ITEMS ...'. A minotaur in view is
mino_guard's (above us in global_strategy).
Targeted pinned replays (19b5f85, KNOWN_ITEMS on vs cand-g): jf45 s14 0.117 -> 0.647 (the blessed scroll at 15/79 HP
next to the mumak, then the castle), jf52 s9 0.075 -> 0.466 (read blind at 27/75 under the soldier ant), jf48 s8
0.126 -> 0.393, jf49 s7 0.379 -> 0.466; jf54 s1 unchanged (5 striking zaps killed the sergeant and a soldier, then the
wand was empty); jf50 s14 unchanged (its wand of fire was known empty).
"""
import nle.nethack as nh

from . import castle_power, jf_config, utils
from . import objects as O
from .combat.monster_utils import WEAK_MONSTERS
from .glyph import G, MON
from .item import Item
from .level import Level
from .strategy import Strategy

GEHENNOM = 1              # dungeon number (dive_logic.GEHENNOM)
RANGED_REACH = 7          # a burst with nothing next to us counts when a hostile is this close (wands, arrows, breath)
_TELE = O.from_name('teleportation', nh.SCROLL_CLASS)
_W = castle_power._W
_NOTELEPORT = 'A mysterious force prevents you from teleporting'


def _cheb(a, b):
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


class KnownItemsGuard:
    def __init__(self, dive, mino):
        self.dive = dive
        self.agent = dive.agent
        self.mino = mino
        self._attempts = {}   # (kind, item glyph) -> (turn, tries in that turn)
        self._blocked = {}    # (kind, item glyph) -> turn until which that action isn't planned
        self.uses = 0         # actions taken (logged)

    # ------------------------------------------------------------------ state

    def _in_scope(self):
        agent = self.agent
        dive = self.dive
        if not dive.diving:
            return False
        level = agent.current_level()
        if level.dungeon_number not in (Level.DUNGEONS_OF_DOOM, Level.GNOMISH_MINES, GEHENNOM):
            return False
        if agent.character.prop.polymorph:
            return False   # a form's HP is a buffer (and castle_cross runs the forms' own plans)
        if utils.any_in(agent.glyphs, G.SWALLOW):
            return False   # engulfed: engulfed_fight
        if dive.levitating() or dive.castle._floating():
            return False   # over the moat: the crossing's business
        return True

    def _danger(self):
        """(adjacent hostiles, 'critical' | 'burst', HP lost) when the guard should act, else None. HP first (cheap),
        then the monsters."""
        agent = self.agent
        bl = agent.blstats
        hp, maxhp = int(bl.hitpoints), int(bl.max_hitpoints)
        critical = agent._critically_low_hp()
        prev = [h for t, h in self.dive._hp_history if t >= bl.time - jf_config.KNOWN_ITEMS_BURST_TURNS]
        loss = (max(prev) - hp) if prev else 0
        burst = loss > 0 and loss >= hp and hp < jf_config.KNOWN_ITEMS_BURST_FRAC * maxhp
        if not (critical or burst):
            return None
        pos = (int(bl.y), int(bl.x))
        monsters = [m for m in agent.get_visible_monsters() if getattr(m[3], 'mname', '') not in WEAK_MONSTERS]
        adjacent = [m for m in monsters if _cheb((m[1], m[2]), pos) <= 1]
        if not adjacent and not (loss > 0 and any(_cheb((m[1], m[2]), pos) <= RANGED_REACH for m in monsters)):
            return None
        if adjacent and self._on_elbereth() and not any(self.dive._melee_ignores_elbereth(m[3]) for m in adjacent) and \
                bl.time - getattr(self.dive, '_hurt_on_elbereth', -100) > 2:
            # an intact Elbereth under us holds everything next to us and nothing has hurt us through it: the danger
            # has passed (da-all2 jf54 s2 zapped striking at a mumak from its fresh Elbereth: 'You feel like a
            # hypocrite. The engraving beneath you fades.' -- mon.c setmangry deletes it -- and the mumak hit again)
            return None
        return adjacent, ('critical' if critical else 'burst'), loss

    def _on_elbereth(self):
        """An Elbereth under us that we can read back (not while blind: dust can't be felt)."""
        agent = self.agent
        return not agent.character.prop.blind and (agent.inventory.engraving_below_me or '').lower() == 'elbereth'

    def _teleport_ok(self):
        """MinoGuard._teleport_ok (castle, Medusa, the Valley, Sokoban, refused before), and not an unrecognised level at
        Medusa's depth showing water (medusa.des: all four variants are noteleport; the dive names the level only
        once it has seen MEDUSA_WET_SQUARES of water)."""
        if not self.mino._teleport_ok():
            return False
        from .dive_logic import MEDUSA_MIN_DEPTH, WET
        agent = self.agent
        level = agent.current_level()
        if level.dungeon_number == Level.DUNGEONS_OF_DOOM and agent.blstats.depth >= MEDUSA_MIN_DEPTH and \
                self.dive.medusa_level is None and utils.isin(level.objects, WET).sum() > 0:
            return False
        return True

    # ------------------------------------------------------------------ items

    def _tele_scroll(self):
        """A known scroll of teleportation we may spend: not known cursed (power_route's level-teleport tickets), not
        WISH_TELEPORT_ROUTE's wished stack, and not with teleport control known (the scroll's 'Where do you want to be
        teleported?' is the TC route's business)."""
        agent = self.agent
        if getattr(agent, '_tele_letter', None) is not None:
            return None
        from . import power_route
        try:
            if power_route.tc_known(agent):
                return None
            reserved = power_route.cursed_tele_scrolls(agent)
        except Exception:
            return None
        for it in agent.inventory.items:
            if it.category == nh.SCROLL_CLASS and it.is_unambiguous() and it.object == _TELE and \
                    it.status != Item.CURSED and it not in reserved:
                return it
        return None

    def _heal_potion(self, critical):
        names = ('full healing', 'extra healing') + (('healing',) if critical else ())
        agent = self.agent
        for name in names:
            for it in agent.inventory.items:
                if it.category == nh.POTION_CLASS and it.is_unambiguous() and it.object.name == name:
                    return it
        return None

    def _unpaid(self):
        """Shop goods in the pack: a teleport out of the shop is theft (the shopkeeper and the Kops come after us)."""
        from .item import flatten_items
        return any(it.shop_status == Item.UNPAID for it in flatten_items(self.agent.inventory.items))

    def _line_hits_peaceful(self, dy, dx):
        """A peaceful (a watchman, a shopkeeper, a Minetown gnome) in view on our zap line within a ray's reach: an
        angry Watch or shopkeeper is worse than what we zap at (fight_heur.missiles_risk_the_watch keeps missiles out
        of Minetown for the same reason)."""
        agent = self.agent
        mask = agent.monster_tracker.peaceful_monster_mask
        h, w = mask.shape
        y, x = int(agent.blstats.y), int(agent.blstats.x)
        for _ in range(13):
            y, x = y + dy, x + dx
            if not (0 <= y < h and 0 <= x < w):
                break
            if mask[y, x]:
                return True
        return False

    def _free_run_behind(self, target, dy, dx):
        """Walkable squares beyond the target in direction (dy, dx) before a wall or the map's edge."""
        level = self.agent.current_level()
        y, x = int(target[1]), int(target[2])
        h, w = level.walkable.shape
        n = 0
        while n < 20:
            y, x = y + dy, x + dx
            if not (0 <= y < h and 0 <= x < w) or not level.walkable[y, x]:
                break
            n += 1
        return n

    # ------------------------------------------------------------------ the plan

    def _candidates(self, adjacent, why, loss):
        agent = self.agent
        dive = self.dive
        mino = self.mino
        bl = agent.blstats
        critical = why == 'critical'
        level = agent.current_level()
        stalker = any(getattr(m[3], 'mflags2', 0) & MON.M2_STALK for m in adjacent)
        # 1. the up staircase under us
        if not stalker and level.dungeon_number != Level.SOKOBAN:
            ups = mino._ups()
            if ups and ups[0][0] == 0:
                yield ('climb', None, 'on the up stairs')
        # 2-3. teleport ourselves (not with shop goods in the pack)
        if self._teleport_ok() and not self._unpaid():
            scroll = self._tele_scroll()
            if scroll is not None:
                yield ('read', scroll, 'known scroll of teleportation')
            wand, _ = mino._wand(('teleportation',))
            if wand is not None:
                yield ('zapself', wand, 'known wand of teleportation at ourselves')
        # 4. a hole
        if not stalker and mino._diggable():
            wand = dive._dig_wand()
            if wand is not None and dive._wand_escape(wand) is not None:
                yield ('zapdown', wand, 'known wand of digging')
        # 5. heal
        potion = self._heal_potion(critical)
        if potion is not None:
            yield ('quaff', potion, 'known healing potion')
        # 6. at the strongest attacker next to us (never in Minetown or with a peaceful on the line)
        if not adjacent:
            return
        from .combat import fight_heur
        if fight_heur.missiles_risk_the_watch(agent):
            return
        target = max(adjacent, key=lambda m: getattr(m[3], 'mlevel', 0))
        if self._on_elbereth() and not dive._melee_ignores_elbereth(target[3]):
            return   # a zap at a monster our Elbereth scares deletes the engraving (mon.c setmangry: 'hypocrite', -5 align)
        pos = (int(bl.y), int(bl.x))
        dy, dx = int(target[1]) - pos[0], int(target[2]) - pos[1]
        dy, dx = (dy > 0) - (dy < 0), (dx > 0) - (dx < 0)
        if self._line_hits_peaceful(dy, dx):
            return
        direction = agent.calc_direction(bl.y, bl.x, int(target[1]), int(target[2]))
        ray_ok = critical or self._free_run_behind(target, dy, dx) >= jf_config.KNOWN_ITEMS_RAY_RUN
        if critical and jf_config.RAY_CRIT_RUN:
            # RAY_CRIT_RUN: at critically low HP a sleep/lightning/magic-missile ray needs a short free run behind the
            # target too. A ray's range is rn1(7,7) squares, -2 per monster it passes (zap.c buzz), so with a wall
            # 1-2 squares behind an adjacent target it comes back through us: a 6d6 bolt or sleep next to the
            # attacker is death at 1/7 HP -- tr0 val-hum-neu-fem s210 ('You kill the leprechaun! The bolt of
            # lightning hits you!'), v2a cav-gno-neu-mal s202 (magic missile at a gelatinous cube). The beam of
            # striking (no bounce) and the other plans stay open.
            ray_ok = self._free_run_behind(target, dy, dx) >= jf_config.RAY_CRIT_RUN
        name = getattr(target[3], 'mname', '?')
        wand, wname = mino._wand(('sleep',))
        if wand is None and critical:
            wand, wname = mino._sleep_or_death(), 'sleep or death'
        if wand is not None and ray_ok:
            yield ('zap', (wand, direction, target), f'known {wname} at the {name}')
        wand, _ = mino._wand(('striking',))
        if wand is not None:
            yield ('zap', (wand, direction, target), f'known striking at the {name}')
        # RAY_BOUNCE_FIX: cold and fire are rays too -- with a wall close behind the target the bolt bounces back
        # through us (6d6): 7 of our dev deaths were our own bounce (a5 arc-dwa s319: cold at an earth elemental).
        # A Valkyrie resists cold; otherwise these need the same free run as the other rays, even when critical.
        safe_run = not jf_config.RAY_BOUNCE_FIX or \
            self._free_run_behind(target, dy, dx) >= jf_config.KNOWN_ITEMS_RAY_RUN
        wand, _ = mino._wand(('cold',))
        if wand is not None and not dive.cold_reserved(wand) and \
                (safe_run or getattr(agent.character, 'role', None) == getattr(agent.character, 'VALKYRIE', -1)):
            yield ('zap', (wand, direction, target), f'known cold at the {name}')
        wand, _ = mino._wand(('fire',))
        if wand is not None and safe_run:
            yield ('zap', (wand, direction, target), f'known fire at the {name}')
        if ray_ok:
            wand, wname = mino._wand(('lightning', 'magic missile'))
            if wand is not None:
                yield ('zap', (wand, direction, target), f'known {wname} at the {name}')

    @staticmethod
    def _block_key(plan):
        kind, arg, _ = plan
        item = arg[0] if isinstance(arg, tuple) and arg and hasattr(arg[0], 'glyphs') else arg
        return (kind, item.glyphs[0] if hasattr(item, 'glyphs') else None)

    def _plan(self):
        """(kind, arg, why, danger) or None. Side-effect free: it runs as the preempt condition on every step."""
        if not jf_config.KNOWN_ITEMS or not self._in_scope():
            return None
        danger = self._danger()
        if danger is None or self.mino._prayer_first():
            return None
        turn = self.agent.blstats.time
        for plan in self._candidates(*danger):
            if self._blocked.get(self._block_key(plan), -1) < turn:
                return plan + (danger,)
        return None

    # ------------------------------------------------------------------ acting

    def _act(self, plan):
        agent = self.agent
        dive = self.dive
        kind, arg, why, (adjacent, trigger, loss) = plan
        bkey = self._block_key(plan[:3])
        t, n = self._attempts.get(bkey, (None, 0))
        n = n + 1 if t == agent.blstats.time else 1
        self._attempts[bkey] = (agent.blstats.time, n)
        if n >= 3:
            # the same action a third time within one game turn: it passes no time (refused) -- no loop on it
            self._blocked[bkey] = agent.blstats.time + 3
            agent.log(f'KNOWN_ITEMS {bkey} refused 3 times at T{agent.blstats.time}: blocked for 3 turns')
        bl = agent.blstats
        key = agent.current_level().key()
        near = [getattr(m[3], 'mname', '?') for m in adjacent[:4]]
        self.uses += 1
        agent.log(f'KNOWN_ITEMS {kind} ({why}): {trigger}, hp {bl.hitpoints}/{bl.max_hitpoints}, lost {loss} in '
                  f'{jf_config.KNOWN_ITEMS_BURST_TURNS} turns, next to us {near}, depth {bl.depth} (use {self.uses})')
        if kind == 'climb':
            agent.move('<')
            return
        if kind == 'read':
            self.mino._read(arg, f'KNOWN_ITEMS: {why}')
            if _NOTELEPORT in (agent.message or ''):
                self.mino._noteleport.add(key)
            return
        if kind == 'zapself':
            agent.zap(arg, '.')
            agent.log(f'KNOWN_ITEMS zap -> {(agent.message or "")[:160]!r}')
            if _NOTELEPORT in (agent.message or ''):
                self.mino._noteleport.add(key)
            agent.inventory.items.update(force=True)
            return
        if kind == 'zapdown':
            dive._escape_act(('zap', arg))
            return
        if kind == 'quaff':
            agent.inventory.quaff(arg)
            agent.log(f'KNOWN_ITEMS quaff -> {(agent.message or "")[:160]!r}')
            return
        if kind == 'zap':
            wand, direction, m = arg
            agent.zap(wand, direction)
            msg = agent.message or ''
            agent.log(f'KNOWN_ITEMS zap -> {msg[:160]!r}')
            if 'Nothing happens' in msg or 'You wrest' in msg:
                agent.inventory.empty_wands.add(wand.text)
            agent.inventory.items.update(force=True)
            return
        raise ValueError(kind)

    def strategy(self):
        def f():
            if not jf_config.KNOWN_ITEMS:
                yield False
                return
            plan = self._plan()
            if plan is None:
                yield False
                return
            yield True
            self._act(plan)

        return Strategy(f)
