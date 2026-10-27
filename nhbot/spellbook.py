"""Observed spell memory and conservative maintenance of usable Wizard spells.

NetHack 3.6 spell.c exposes retention intervals, not exact remaining turns.
Only their upper bound can justify refreshing at the 2000-turn threshold.
"""
from dataclasses import dataclass, replace
import re

import nle.nethack as nh
from nle.nethack import actions as A

from .glyph import Hunger
from .item import Item
from .strategy import Strategy


USABLE_SPELLS = ('force bolt', 'healing', 'extra healing')
_ROW = re.compile(r'^([a-zA-Z]) -\s+(.+?)\s+(\d+)\s+(\w+)\s+(\d+)%\s+'
                  r'(\(gone\)|\d+%(?:-\d+%)?)\s*$')


@dataclass(frozen=True)
class SpellKnowledge:
    letter: str
    name: str
    level: int
    category: str
    failure: float
    retention: tuple
    observed_turn: int
    expires_by: int = None

    def remaining_upper(self, turn):
        expiry = (self.observed_turn + self.retention[1] * 200
                  if self.expires_by is None else self.expires_by)
        return max(0, expiry - max(turn, self.observed_turn))


def refine_spell_knowledge(previous, observed):
    """Intersect menu evidence without resetting a broad interval's clock.

    A higher lower bound proves replenishment and starts a fresh estimate.
    Retention's displayed N% lower endpoint means strictly greater than
    (N-1)%, per spell.c; a menu saying 1%-25% includes one remaining turn.
    """
    if previous is None or previous.name != observed.name or observed.retention == (0, 0):
        return observed
    if observed.observed_turn < previous.observed_turn:
        return observed
    lower = (observed.retention[0] - 1) * 200 + 1
    if observed.retention == (100, 100):
        lower = 20000
    previous_upper = previous.remaining_upper(observed.observed_turn)
    if lower > previous_upper:
        return observed
    return replace(observed, expires_by=observed.observed_turn + min(
        previous_upper, observed.remaining_upper(observed.observed_turn)))


def parse_spell_row(line, turn):
    match = _ROW.fullmatch(line)
    if match is None:
        raise ValueError('Invalid spell menu row: ' + repr(line))
    letter, name, level, category, fail, memory = match.groups()
    if memory == '(gone)':
        interval = (0, 0)
    else:
        ends = tuple(int(v.rstrip('%')) for v in memory.split('-'))
        interval = (ends[0], ends[-1])
    if not 0 <= interval[0] <= interval[1] <= 100 or not 0 <= int(fail) <= 100:
        raise ValueError('Invalid spell probability or retention: ' + repr(line))
    return SpellKnowledge(letter, name, int(level), category, int(fail) / 100, interval, int(turn))


def usable_spells(agent=None):
    from .utility_spells import enabled, UTILITY_SPELLS
    return USABLE_SPELLS + (UTILITY_SPELLS if agent is not None and enabled(agent) else ())


def useful_book(item, agent=None):
    return (item.category == nh.SPBOOK_CLASS and item.is_unambiguous()
            and item.object.name in usable_spells(agent) and item.shop_status == Item.NOT_SHOP
            and item.status in (Item.BLESSED, Item.UNCURSED))


def needs_study(character, item, turn):
    catalog = getattr(character, 'spell_catalog', None)
    if catalog is None:  # a refused menu is not evidence that a spell is unknown
        return False
    spell = catalog.get(item.object.name)
    return spell is None or spell.remaining_upper(turn) <= 2000


def safe_to_read(agent, item):
    if not useful_book(item, agent):
        return False
    if item.object.name not in USABLE_SPELLS:
        from .utility_spells import FOOD_RESERVE
        from .item.utils import public_noncursed
        # ItemManager's legacy default UNCursed status is not public BUC
        # evidence. Newly enabled long studies must not read a possibly cursed
        # book merely because its type and comprehension chance are known.
        if not public_noncursed(item):
            return False
        if agent.inventory.carried_nutrition() < FOOD_RESERVE:
            return False
        if item.object.name == 'detect monsters' and not (
                'identify' in (agent.character.spell_catalog or {}) or
                any(it.is_unambiguous() and it.object.name == 'identify' and
                    useful_book(it, agent) for it in agent.inventory.items)):
            return False
    # Blessed books bypass the comprehension roll. For uncursed books require
    # guaranteed success without assuming the hero has lenses (spell.c).
    return (item.status == Item.BLESSED or
            agent.blstats.intelligence + 4 + agent.blstats.experience_level // 2
            - 2 * item.object.level >= 20)


@Strategy.wrap
def study(agent):
    character, bl = agent.character, agent.blstats
    prop = character.prop
    if (character.role != character.WIZARD or
            any((prop.blind, prop.confusion, prop.stun, prop.hallu, prop.polymorph)) or
            bl.hunger_state > Hunger.NOT_HUNGRY or bl.hitpoints != bl.max_hitpoints or
            bl.carrying_capacity or agent.get_visible_monsters() or
            agent.current_level().shop[bl.y, bl.x] or agent.hands_welded()):
        yield False
    blocked = getattr(agent, '_spellbook_retry_after', {})
    books = [it for it in agent.inventory.items if safe_to_read(agent, it)
             and needs_study(character, it, bl.time)
             and bl.time >= blocked.get(it.object.name, -1)]
    if not books:
        yield False
    book = min(books, key=lambda it: usable_spells(agent).index(it.object.name))
    yield True
    # Set before acting so interrupted or refused study cannot monopolize the
    # scheduler. A later observation must confirm successful learning.
    agent._spellbook_retry_after = blocked
    blocked[book.object.name] = int(bl.time) + 250
    letter = agent.inventory.items.get_letter(book)

    def answers():
        if 'What do you want to read?' not in agent.single_message:
            return
        yield letter
        if 'difficult to comprehend' in agent.single_message:
            yield 'n'  # changed state or a mistaken estimate must fail closed

    agent.log(f'SPELLBOOK studying {book.text!r}')
    with agent.atom_operation():
        agent.step(A.Command.READ, answers())
    character.parse_spellcast_view()
    agent.inventory.items.update(force=True)
