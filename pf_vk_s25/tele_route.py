"""WISH_TELEPORT_ROUTE: a wand of wishing -> the bottom of Gehennom by two controlled level teleports.

teleport.c level_tele(): with teleport control, any level asked for past the bottom of the Dungeons of Doom is
find_hell() -- the Valley of the Dead, castle depth + 1 -- from ANY level of the Dungeons, Dlvl 1 included; from
Gehennom a big number clamps to the vibrating-square level (bottom-1; the Sanctum needs the invocation), Dlvl
~44-52. read.c seffects(): a cursed (or confused) scroll of teleportation calls level_tele(), and noteleport
levels don't stop it (only Sokoban, the endgame and carrying the Amulet do). LEVELPORT_DEEP answers the prompt
with 50 (the top of the progress table). So a ring of teleport control plus two cursed scrolls of teleportation
score 0.78-0.81 wherever we are,
past Medusa, the castle and the Valley in two turns.

A wand of wishing always gives two wishes: the engrave-test wish, rnd(3) - 1 more charges, and one wrested
charge (zap.c zappable(): 1 chance in 121 per zap at 0 charges; the wand then turns to dust). Dev evidence:
base3-jf14 s14 and s8, base4arm/base5arm-jf16 s5 found a wand of wishing on Dlvl 1-3 (2-4 wishes, spent on gray
dragon scale mail and life saving) and died at Dlvl 25-28.

WISH_CHARGING_FIRST: the first wish from the wand is '2 blessed scrolls of charging'. The route's wishes zap the wand
down to (x:0) (the first 'Nothing happens' tells an unknown count), then one scroll is read on it: read.c recharge()
gives a never-recharged wand of wishing 3 charges (blessed; a second recharge explodes it), i.e. c + 2 wishes in all.

  route_wish(agent)                  the next wish the route still needs, or None
  note_wished(agent, text, letter)   remember where the wished (cursed) scrolls went
  teleport_route_strategy(agent)     put the ring on, wrest the wand, read the scrolls
"""

import re

import nle.nethack as nh
from nle.nethack import actions as A

from . import jf_config
from . import objects as O
from .strategy import Strategy

TC_RING = O.from_name('teleport control', nh.RING_CLASS)
TELE_SCROLL = O.from_name('teleportation', nh.SCROLL_CLASS)
WISH_WAND = O.from_name('wishing', nh.WAND_CLASS)
WISH_TC_RING = 'blessed ring of teleport control'
# readobjnam: a count of 2 is granted when 2 < rnd(6) (4 times in 6), else 1; 3 would be granted only half the time
WISH_TELE_SCROLLS = '2 cursed scrolls of teleportation'
# WISH_CHARGING_FIRST: one scroll recharges the wand; the second is a spare (a lost one, the castle's own wand)
CHARGING_SCROLL = O.from_name('charging', nh.SCROLL_CLASS)
WISH_CHARGING = '2 blessed scrolls of charging'
# ROUTE_GLOVES_FIX: do_wear.c puts a ring on under uncursed gloves, but cursed gloves ('You cannot remove your gloves
# to put on the ring.') or a welded weapon block it without using a move -- the ring step looped (front-strong
# fs7-k6 s9: 385k steps after the castle wand). A blessed scroll of remove curse uncurses the whole pack.
REMOVE_CURSE = O.from_name('remove curse', nh.SCROLL_CLASS)
WISH_REMOVE_CURSE = 'blessed scroll of remove curse'
PUTON_BLOCKED = ('cannot remove your gloves', 'cannot free your weapon hand')
MAX_WREST_ZAPS = 500       # P(no wrest in 500 zaps) = (120/121)^500 ~ 1.6%

DUNGEONS_OF_DOOM, GEHENNOM = 0, 1


def _items(agent):
    return list(agent.inventory.items)


def tc_ring(agent):
    return next((i for i in _items(agent) if i.is_unambiguous() and i.object == TC_RING), None)


def wishing_wand(agent):
    """A wand of wishing that can still give a wish: not a cancelled one ('(0:-1)': zappable() never wrests from
    spe < 0, so zapping it would only burn MAX_WREST_ZAPS turns)."""
    return next((i for i in _items(agent) if i.is_unambiguous() and i.object == WISH_WAND and
                 not re.search(r'\(\d+:-\d+\)', i.text or '')), None)


def tele_scrolls(agent):
    """The stack the scroll wishes went to (they are cursed; the inventory doesn't show it: bknown is unset)."""
    letter = getattr(agent, '_tele_letter', None)
    if letter is None:
        return None
    for it in _items(agent):
        if agent.inventory.items.get_letter(it) == letter and it.category == nh.SCROLL_CLASS:
            return it
    return None


def charging_scroll(agent):
    """WISH_CHARGING_FIRST: the stack the charging wish went to (blessed unless Luck < 0; bknown is unset)."""
    letter = getattr(agent, '_charging_letter', None)
    if letter is None:
        return None
    for it in _items(agent):
        if agent.inventory.items.get_letter(it) == letter and it.category == nh.SCROLL_CLASS:
            return it
    return None


def _recharged(agent, wand):
    """The wand was recharged: by us, or its known count says so ('(1:3)'). A second recharge explodes it."""
    if getattr(agent, '_wish_recharged', False):
        return True
    m = re.search(r'\((\d+):-?\d+\)', wand.text or '') if wand is not None else None
    return m is not None and int(m.group(1)) > 0


def _known_empty(agent, wand):
    """A zap said 'Nothing happens' (agent.zap: inventory.empty_wands), or the known count is 0."""
    return agent.inventory.is_known_empty(wand) or bool(re.search(r'\(\d+:0\)', wand.text or ''))


def _charging_wanted(agent):
    """WISH_CHARGING_FIRST: the next wand wish is the charging scrolls (asked once; a wish for them that failed --
    Luck, a full pack -- leaves the old route)."""
    if not jf_config.WISH_CHARGING_FIRST or getattr(agent, '_charging_asked', False) or \
            charging_scroll(agent) is not None or not _wand_source(agent):
        return False
    return not _recharged(agent, wishing_wand(agent))


def note_asked(agent, text):
    """The wish prompt was answered with `text` (power.note_wish)."""
    if text == WISH_CHARGING:
        agent._charging_asked = True
    elif text == WISH_REMOVE_CURSE:
        agent._remove_curse_asked = True


def _wand_source(agent):
    """The wish comes from a wand of wishing (an engrave-test or a zap just now, or one we know): >= 2 wishes. A
    throne, fountain or lamp gives one, and a ring of teleport control alone scores nothing."""
    return wishing_wand(agent) is not None or \
        agent.step_count - getattr(agent, '_last_wand_use_step', -10 ** 9) <= 60


def _tc_known(agent):
    """A TC ring we wished/identified, or (TC_ROUTE) teleport control learned any other way: the intrinsic from a
    tengu, a ring the game's TC prompts proved."""
    if tc_ring(agent) is not None:
        return True
    if jf_config.TC_ROUTE:
        from . import power_route   # (power_route imports power, which imports this module)
        return power_route.tc_known(agent)
    return False


def route_wish(agent):
    if not jf_config.WISH_TELEPORT_ROUTE or getattr(agent, '_tele_route_done', False):
        return None
    if _charging_wanted(agent):
        return WISH_CHARGING
    if not _tc_known(agent):
        # the ring is worth a wish when more wishes follow (a wand: the scrolls come next) -- and with TC_ROUTE even
        # a single one (fountain demon, throne, lamp): 9 of 12 revealed castle kits carry teleport scrolls, 1 in 8
        # of them cursed, which the castle gamble / identify / altar tests turn into jumps; power-route's estimate
        # of P(score > 0.647): TC ring 8-10%, levitation ring ~5%, gray dragon scale mail ~1%
        return WISH_TC_RING if (_wand_source(agent) or jf_config.TC_ROUTE) else None
    if jf_config.ROUTE_GLOVES_FIX and getattr(agent, '_tele_puton_blocked', False) and \
            not getattr(agent, '_remove_curse_asked', False) and remove_curse_scroll(agent) is None:
        return WISH_REMOVE_CURSE
    scrolls = tele_scrolls(agent)
    if scrolls is None or scrolls.count < 2:
        return WISH_TELE_SCROLLS
    return None


def remove_curse_scroll(agent):
    """ROUTE_GLOVES_FIX: the stack the remove curse wish went to."""
    letter = getattr(agent, '_remove_curse_letter', None)
    if letter is None:
        return None
    for it in _items(agent):
        if agent.inventory.items.get_letter(it) == letter and it.category == nh.SCROLL_CLASS:
            return it
    return None


def note_wished(agent, text, letter):
    if text == WISH_TELE_SCROLLS:
        agent._tele_letter = letter
        agent.log(f'TELEPORT route: cursed scrolls of teleportation at {letter!r}')
    elif text == WISH_CHARGING:
        agent._charging_letter = letter
        agent.log(f'TELEPORT route: blessed scrolls of charging at {letter!r}')
    elif text == WISH_REMOVE_CURSE:
        agent._remove_curse_letter = letter
        agent.log(f'TELEPORT route: blessed remove curse at {letter!r}')


def _read_charging(agent, wand, scroll):
    """Read the charging scroll on the wand ('This is a charging scroll.' -> 'What do you want to charge?')."""
    sl = agent.inventory.items.get_letter(scroll)
    wl = agent.inventory.items.get_letter(wand)
    agent.log(f'TELEPORT route: reading {scroll.text!r} ({sl}) on {wand.text!r} ({wl})')
    agent._wish_recharged = True     # whatever happens: a second recharge would explode the wand
    with agent.atom_operation():
        agent.step(A.Command.READ)
        agent.type_text(sl)
        if 'What do you want to charge?' in agent.single_message:
            agent.type_text(wl)
        if 'What do you want to charge?' in agent.single_message:
            agent.step(A.Command.ESC)
    agent.log(f'TELEPORT route: charging -> {agent.message[:160]!r}')
    agent.inventory.empty_wands.discard(wand.text)
    agent._tele_zaps = 0
    agent.inventory.items.update(force=True)


def _wallwalker_ok(agent):
    """CASTLE_TREASURY: a wall-walking form runs the route too -- the xorn that took the castle's wand of wishing has
    hands, eyes and fingers (monst.c: no M1_NOHANDS/NOEYES/NOLIMBS), so it zaps, reads and puts the ring on, and the
    Valley below is where its form would run out before the route could start."""
    if not jf_config.CASTLE_TREASURY:
        return False
    try:
        from . import castle_cross
        return castle_cross.wallwalker(agent)
    except Exception:
        return False


def teleport_route_strategy(agent):
    def f():
        if not jf_config.WISH_TELEPORT_ROUTE or getattr(agent, '_tele_route_done', False) or \
                (agent.character.prop.polymorph and not _wallwalker_ok(agent)):
            yield False
            return
        bl = agent.blstats
        ring = tc_ring(agent)
        scrolls = tele_scrolls(agent)
        wand = wishing_wand(agent)
        if ring is None and scrolls is None and wand is None:
            yield False
            return
        near = [m for m in agent.get_visible_monsters() if max(abs(m[1] - bl.y), abs(m[2] - bl.x)) <= 3]

        # 1) wear the ring of teleport control (the prompt needs it on; a cursed one staying on is fine)
        blocked = jf_config.ROUTE_GLOVES_FIX and getattr(agent, '_tele_puton_blocked', False)
        if blocked and ring is not None and not ring.equipped and remove_curse_scroll(agent) is None and \
                getattr(agent, '_remove_curse_asked', False) and not getattr(agent, '_puton_retried', False):
            # the scroll is gone (read by us or by another strategy -- power_route's castle gamble reads unknown
            # scrolls): try the ring once more
            agent._puton_retried = True
            agent._tele_puton_blocked = blocked = False
        if ring is not None and not ring.equipped and blocked:
            rc = remove_curse_scroll(agent)
            if rc is not None:
                yield True
                rl = agent.inventory.items.get_letter(rc)
                agent.log(f'TELEPORT route: reading {rc.text!r} ({rl}): the ring is blocked')
                with agent.atom_operation():
                    agent.step(A.Command.READ)
                    agent.type_text(rl)
                agent.log(f'TELEPORT route: remove curse -> {agent.message[:120]!r}')
                agent._tele_puton_blocked = False
                agent._remove_curse_letter = None
                agent.inventory.items.update(force=True)
                return
            if route_wish(agent) != WISH_REMOVE_CURSE or wand is None:
                yield False   # nothing left to lift the curse: no loop on a put-on that takes no time
                return
            # else: on to step 2, which zaps the wand for the remove curse
        elif ring is not None and not ring.equipped:
            worn = [i for i in _items(agent) if i.category == nh.RING_CLASS and i.equipped]
            if len(worn) >= 2:
                agent.log('TELEPORT route: both ring fingers busy, cannot put the ring of teleport control on')
                yield False
                return
            yield True
            letter = agent.inventory.items.get_letter(ring)

            def gen():
                if 'What do you want to put on?' not in agent.single_message:
                    return
                yield letter
                if 'Which ring-finger' in agent.single_message:
                    yield 'r'

            agent.log(f'TELEPORT route: putting on {ring.text!r}')
            with agent.atom_operation():
                agent.step(A.Command.PUTON, gen())
            if jf_config.ROUTE_GLOVES_FIX and any(w in (agent.message or '') for w in PUTON_BLOCKED):
                agent.log(f'TELEPORT route: the ring is blocked: {agent.message[:120]!r}')
                agent._tele_puton_blocked = True
            agent.inventory.items.update(force=True)
            return

        # 2) the wishes the route still needs: zap the wand, wresting its last charge when it is empty
        need = route_wish(agent)
        # WISH_CHARGING_FIRST: at (x:0), before any wrest (a wrest turns the wand to dust), read a charging scroll
        charging = charging_scroll(agent) if jf_config.WISH_CHARGING_FIRST else None
        if charging is not None and need is not None and wand is not None and not near and \
                _known_empty(agent, wand) and not _recharged(agent, wand):
            prop = agent.character.prop
            if not (prop.stun or prop.confusion or prop.blind or prop.hallu):
                # (confused, the scroll charges our energy instead)
                yield True
                _read_charging(agent, wand, charging)
                return
        if need is not None and wand is not None and not near:
            zaps = getattr(agent, '_tele_zaps', 0)
            if zaps < MAX_WREST_ZAPS:
                yield True
                agent._tele_zaps = zaps + 1
                if zaps % 50 == 0:
                    agent.log(f'TELEPORT route: zapping {wand.text!r} for {need!r} (zap {zaps + 1})')
                agent.zap(wand, None)
                agent.inventory.items.update(force=True)
                return

        # 3) read: in the Dungeons once two scrolls are in hand (or no more wishes can come), then once in Gehennom
        if scrolls is None or ring is None or not ring.equipped:
            yield False
            return
        dnum = bl.dungeon_number
        prop = agent.character.prop
        if dnum not in (DUNGEONS_OF_DOOM, GEHENNOM) or prop.stun or prop.confusion or prop.blind or \
                prop.hallu:
            yield False   # Sokoban/Mines/Quest: no controlled trip to Gehennom; stunned or confused: a random one
            return
        if dnum == GEHENNOM and getattr(agent, '_tele_route_gehennom', False):
            yield False
            return
        if dnum == DUNGEONS_OF_DOOM and scrolls.count < 2 and need is not None and wand is not None and \
                getattr(agent, '_tele_zaps', 0) < MAX_WREST_ZAPS:
            yield False   # another wish is coming: go through both levels in one go
            return
        yield True
        key = (bl.dungeon_number, bl.level_number)
        letter = agent.inventory.items.get_letter(scrolls)
        agent.log(f'TELEPORT route: reading {scrolls.text!r} ({letter}) on {key}, depth {bl.depth}')
        with agent.atom_operation():
            agent.step(A.Command.READ)
            agent.type_text(letter)
        agent.inventory.items.update(force=True)
        after = agent.blstats
        agent.log(f'TELEPORT route: now on {(after.dungeon_number, after.level_number)} depth {after.depth}: '
                  f'{agent.message[:120]!r}')
        if dnum == GEHENNOM:
            agent._tele_route_gehennom = True
            agent._tele_route_done = True

    return Strategy(f)
