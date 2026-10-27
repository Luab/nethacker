"""Conservative use of owned enchant-armor scrolls during Wizard preparation.

NetHack 3.6.7 read.c selects some_armor() without asking for a target. Its
ordinary destruction check uses the existing enchantment: every possible
worn target must therefore be known at +3 or below before a noncursed read.
This action neither changes equipment nor enables the ARMOR_UP scheduler.
"""
import nle.nethack as nh
from nle.nethack import actions as A

from .glyph import Hunger
from .item import Item
from .item.utils import public_noncursed
from .strategy import Strategy


RETRY_TURNS = 100
MAX_UNCHANGED_ATTEMPTS = 2


def _safe_scroll(item):
    return (item.category == nh.SCROLL_CLASS and item.is_unambiguous() and
            item.object.name == 'enchant armor' and
            public_noncursed(item) and
            item.shop_status == Item.NOT_SHOP and item.count > 0)


def _worn(items):
    armor = [item for item in items if item.is_armor() and item.equipped]
    # InventoryItems' slots use Item.equipped, which also means wielded.
    # Require the observed wearing annotation rather than enchanting a
    # cloak merely wielded in the weapon slot. Unrecognized text fails closed.
    if not armor or any('(being worn' not in (item.text or '') for item in armor):
        return None
    return armor


def _safe_worn(items):
    armor = _worn(items)
    if armor is None or any(item.modifier is None or item.modifier > 3 or
                            item.shop_status != Item.NOT_SHOP for item in armor):
        return None
    return armor


def _armor_state(items):
    return tuple(sorted((items.get_letter(item), tuple(obj.name for obj in item.objs), item.modifier)
                        for item in items if item.is_armor() and item.equipped))


def _scroll_count(items):
    return sum(item.count for item in items if _safe_scroll(item))


def _choice(agent):
    character, bl = agent.character, agent.blstats
    prop = character.prop
    if (character.role != character.WIZARD or
            getattr(agent.inventory, '_armor_fitting', None) is not None or
            any((prop.blind, prop.confusion, prop.stun, prop.hallu, prop.polymorph)) or
            bl.hunger_state > Hunger.NOT_HUNGRY or bl.carrying_capacity >= 2 or
            agent.get_visible_monsters()):
        return None
    items = agent.inventory.items  # top-level inventory only, never bag contents
    if _safe_worn(items) is None:
        return None
    scrolls = [item for item in items if _safe_scroll(item)]
    if not scrolls:
        return None
    signature = (_armor_state(items), tuple((items.get_letter(item), item.status, item.count)
                                          for item in scrolls))
    previous = getattr(agent, '_armor_maintenance', None)
    if previous is not None and (bl.time < previous['retry_after'] or
            (signature == previous['signature'] and previous['attempts'] >= MAX_UNCHANGED_ATTEMPTS)):
        return None
    return scrolls[0], signature


@Strategy.wrap
def enchant_owned_armor(agent):
    if _choice(agent) is None:
        yield False
        return
    yield True
    # A strategy can be selected and then preempted before it acts. Re-read
    # ownership, worn enchantments, hunger and threat state before using a scroll.
    choice = _choice(agent)
    if choice is None:
        return
    scroll, signature = choice
    previous = getattr(agent, '_armor_maintenance', None)
    attempts = previous['attempts'] if previous is not None and previous['signature'] == signature else 0
    state = dict(signature=signature, attempts=attempts + 1,
                 retry_after=int(agent.blstats.time) + RETRY_TURNS,
                 outcome='attempting', selected=False,
                 armor_before=_armor_state(agent.inventory.items),
                 scrolls_before=_scroll_count(agent.inventory.items),
                 ac_before=int(agent.blstats.armor_class))
    # Persist before any step, including an exception or preemption at READ.
    agent._armor_maintenance = state
    letter = agent.inventory.items.get_letter(scroll)

    def answers():
        if 'What do you want to read?' in agent.single_message:
            state['selected'] = True
            yield letter

    agent.log(f'ARMOR_MAINT reading {scroll.text!r} attempt={state["attempts"]}')
    try:
        with agent.atom_operation():
            agent.step(A.Command.READ, answers())
        # Refresh from the observed inventory if it changed. force=True would
        # also reopen every container, beyond the scope of this maintenance.
        agent.inventory.items.update()
        after = _armor_state(agent.inventory.items)
        before_by_letter = {letter: (names, modifier) for letter, names, modifier in state['armor_before']}
        increased = any(letter in before_by_letter and names == before_by_letter[letter][0] and
                        modifier is not None and modifier > before_by_letter[letter][1]
                        for letter, names, modifier in after)
        state['ac_after'] = int(agent.blstats.armor_class)
        state['consumed'] = _scroll_count(agent.inventory.items) < state['scrolls_before']
        if increased or state['ac_after'] < state['ac_before']:
            state['outcome'] = 'observed_improvement'
        elif state['consumed']:
            state['outcome'] = 'consumed_without_observed_improvement'
        else:
            state['outcome'] = 'no_observed_change'
    finally:
        if state['outcome'] == 'attempting':
            state['outcome'] = 'interrupted'
        agent.log(f'ARMOR_MAINT outcome={state["outcome"]} selected={state["selected"]} '
                  f'ac_before={state["ac_before"]} ac_after={state.get("ac_after")} '
                  f'consumed={state.get("consumed")} retry_after={state["retry_after"]}')
