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
    if not _tc_known(agent):
        # the ring is worth a wish when more wishes follow (a wand: the scrolls come next) -- and with TC_ROUTE even
        # a single one (fountain demon, throne, lamp): 9 of 12 revealed castle kits carry teleport scrolls, 1 in 8
        # of them cursed, which the castle gamble / identify / altar tests turn into jumps; power-route's estimate
        # of P(score > 0.647): TC ring 8-10%, levitation ring ~5%, gray dragon scale mail ~1%
        return WISH_TC_RING if (_wand_source(agent) or jf_config.TC_ROUTE) else None
    scrolls = tele_scrolls(agent)
    if scrolls is None or scrolls.count < 2:
        return WISH_TELE_SCROLLS
    return None


def note_wished(agent, text, letter):
    if text == WISH_TELE_SCROLLS:
        agent._tele_letter = letter
        agent.log(f'TELEPORT route: cursed scrolls of teleportation at {letter!r}')


def teleport_route_strategy(agent):
    def f():
        if not jf_config.WISH_TELEPORT_ROUTE or getattr(agent, '_tele_route_done', False) or \
                agent.character.prop.polymorph:
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
        if ring is not None and not ring.equipped:
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
            agent.inventory.items.update(force=True)
            return

        # 2) the wishes the route still needs: zap the wand, wresting its last charge when it is empty
        need = route_wish(agent)
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
